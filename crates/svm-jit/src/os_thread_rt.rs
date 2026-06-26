//! **1:1 OS-thread executor** for the JIT (§12) — the VM exposes `thread.spawn`/`thread.join` and the
//! `wait`/`notify` futex as *primitives*, **not** a scheduler. A spawned vCPU is one real OS thread
//! (DESIGN §12: "vCPU — a capability to run on a physical core … an OS thread the host scheduler
//! runs"); the guest runtime builds whatever threading model it wants (M:N green threads, pools,
//! priorities) on top of these + `cont.*` fibers. There is **no green-thread multiplexing and no
//! scheduling policy in the VM** (D22: "no built-in scheduler / no double-scheduling").
//!
//! Shape: the root vCPU (`main`) runs on the caller's thread under the §5 `run_guarded` shim; each
//! `thread.spawn` launches a fresh OS thread running the guest entry under that same per-OS-thread
//! `setjmp`/`siglongjmp` guard ([`mem::run_guarded_range`]). `thread.join` blocks the calling OS
//! thread on the child's completion cell; `wait`/`notify` is a host-side futex over confined guest
//! addresses. All vCPUs share the one guest window → real hardware atomics. `run_inner` joins every
//! spawned OS thread before tearing the run down, so nothing outlives the window/code.
//!
//! Detect-and-kill (§5): a guest memory fault on any vCPU `siglongjmp`s out of that thread's guarded
//! call; the thread records the trap in its completion cell + the shared trap cell, and the joiner
//! propagates it. The **fuel/epoch kill-path** for a *runaway* (non-faulting) guest works across the
//! whole domain: the lowering polls a host-owned interrupt cell at loop back-edges + function entries
//! and traps `OutOfFuel` when the host sets it (see `compile_and_run_with_host_interruptible` /
//! `emit_epoch_check`). Because every vCPU runs the same finalized code, a *spinning* sibling polls
//! that one cell on its own; a *parked* sibling (blocked in a futex `wait` or `thread.join`) re-checks
//! it on a bounded interval (`KILL_RECHECK`, real-build only) so it wakes and unwinds too — so a
//! single host interrupt stops the entire multithreaded domain, not just its busy threads, and
//! `join_all` never hangs on a vCPU that would otherwise wait forever.

use crate::fiber_rt::{self, FiberCallTramp, FiberRuntime, SharedFiberTable};
use crate::{mem, FnEntry, TrapKind};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// loom swaps the synchronization primitives for its model-checked versions; `Arc` and `JoinHandle`
// stay std (loom doesn't track them, and the real spawn path is `cfg(not(loom))`). The futex core is
// exercised by the loom test below via loom threads.
#[cfg(loom)]
use loom::sync::{Condvar, Mutex};
#[cfg(not(loom))]
use std::sync::{Condvar, Mutex};

/// `<ty>.atomic.wait` status results (§12, matching the interpreter / wasm).
const WAIT_WOKEN: i32 = 0;
const WAIT_NOT_EQUAL: i32 = 1;
// Only produced on the timed-wait path, which loom (no timeout model) compiles out.
#[cfg_attr(loom, allow(dead_code))]
const WAIT_TIMED_OUT: i32 = 2;
/// §12.8 concurrent-thaw stage 2: an **internal** status — an infinite (no-timeout) wait whose every
/// possible notifier has exited (`peers_live()` is false), so it can never be satisfied. The
/// `atomic.wait` thunk intercepts it and traps `ThreadFault` (matching the interpreter, which surfaces a
/// guest wait/join-deadlock as `Trap::ThreadFault`); it is **never** returned to the guest as a status.
/// Lets the blanket thaw fail-closed go away: a re-issued wait now parks while a live sibling could
/// notify (so a producer↔consumer pair frozen mid-rendezvous resolves) and fails closed only here, when
/// no peer remains — without hanging. The loom model exercises this directly (see the deadlock test).
const WAIT_DEADLOCK: i32 = 3;

/// Max concurrently-live vCPUs per run (matches the interpreter's `MAX_VCPUS`): an anti-bomb ceiling
/// so a thread-bomb traps (`ThreadFault`) instead of exhausting host memory.
const MAX_VCPUS: usize = 1 << 16;

/// How often a *parked* vCPU (futex `wait` / `thread.join`) re-checks the §5 kill-path interrupt
/// cell when the kill-path is armed. A spinning vCPU trips near-instantly (the cell is polled at
/// every back-edge); this only bounds the extra latency for an otherwise-blocked vCPU to unwind
/// once the host requests a kill. Real-build only (the loom futex model uses no timeouts and never
/// arms the kill-path).
#[cfg(not(loom))]
const KILL_RECHECK: Duration = Duration::from_millis(20);

/// Atomically store a trap code into the run's shared trap cell (audit #2). The cell is one `i64`
/// shared by every vCPU; multiple dying vCPUs (and the root) can store to it concurrently, so the
/// **Rust** accesses must be atomic to avoid a data race. `p` points at the run's `AtomicI64` storage
/// (`run_inner`); the JIT writes the same cell via an aligned `i64` store in its emitted code (a
/// hardware-atomic store, foreign to Rust's abstract machine). Relaxed suffices: any trap value is
/// terminal and the final read (post-`join_all`, a synchronization point) sees the last store.
///
/// # Safety
/// `p` is the run's live trap-cell address (an `AtomicI64`'s storage), valid for the whole run.
unsafe fn store_trap(p: *mut i64, v: i64) {
    (*(p as *const AtomicI64)).store(v, Ordering::Relaxed);
}
/// Atomically load the run's shared trap cell (audit #2; see [`store_trap`]).
///
/// # Safety
/// As [`store_trap`].
unsafe fn load_trap(p: *mut i64) -> i64 {
    (*(p as *const AtomicI64)).load(Ordering::Relaxed)
}

/// True iff the kill-path is armed (`addr != 0`) **and** the host has set the interrupt cell — i.e.
/// a parked vCPU should stop waiting and unwind (its next epoch poll in guest code traps `OutOfFuel`).
fn epoch_fired(addr: usize) -> bool {
    // SAFETY: a non-zero `addr` is the run's live interrupt cell (an `AtomicU64`); it outlives every
    // vCPU (joined before `run_inner` frees it). A relaxed load suffices — we only need eventual
    // visibility of the host's store, and the kill is idempotent.
    addr != 0 && unsafe { (*(addr as *const AtomicU64)).load(Ordering::Relaxed) != 0 }
}

/// Per-run constants every vCPU needs to call guest code — constant for the whole run, copied into
/// each spawned thread.
#[derive(Clone, Copy)]
struct Env {
    mem_base: u64,
    fn_table_base: u64,
    trap_out: *mut i64,
    call_tramp: FiberCallTramp,
    fault_lo: usize,
    fault_hi: usize,
    /// `(fiber_type_id, fn_table_mask)` when the module mixes fibers + threads — each vCPU then gets
    /// its own `FiberRuntime` (execution context over the domain-shared [`SharedFiberTable`], held
    /// by the `Domain`) for `cont.*`. `None` for pure-thread modules.
    fiber_cfg: Option<(u32, u64)>,
    /// Address of the §5 kill-path interrupt cell (an `AtomicU64`), or `0` when no kill-path is
    /// armed. *Spinning* vCPUs already poll it (it is baked into the same compiled code every vCPU
    /// runs); this lets a **parked** vCPU — blocked in a futex `wait` or `thread.join` — re-check it
    /// and unwind too, so one host interrupt kills the whole domain rather than only its busy threads.
    epoch_addr: usize,
    /// This is a **durable** (freeze/thaw) run. When the window state ≠ NORMAL (a freeze/thaw in
    /// progress), `thread.spawn` runs the child **inline** on the spawning thread (single-worker,
    /// mirroring the interp), so the shared durable control words are never raced (slice 3.3).
    durable: bool,
}

// SAFETY: `Env`'s raw pointers refer to the run's shared window / trap cell, valid for the whole run.
// The trap cell is only ever written (idempotently, to a `TrapKind`) when the domain is being killed,
// so racy writes from multiple dying vCPUs are sound; the window is the shared guest memory.
unsafe impl Send for Env {}
unsafe impl Sync for Env {}

/// A spawned vCPU's completion cell: its `i64` result once finished, and the trap code (`0` = clean).
struct Done {
    state: Mutex<Option<(i64, i64)>>,
    cv: Condvar,
}

/// The per-run thread table + futex — the "scheduler address" baked into the `thread.*` thunks. It
/// owns **no scheduling policy**: spawn = OS thread, join = wait on a completion cell, wait/notify =
/// futex. Created before code is finalized (to bake its address); [`Domain::set_env`] supplies the
/// call-trampoline-bearing [`Env`] afterwards.
pub(crate) struct Domain {
    env: Mutex<Option<Env>>,
    /// The **domain-shared fiber table** (D57 3b-ii) every spawned vCPU's `FiberRuntime` is built
    /// over — one handle namespace + one §15 fiber quota for the whole domain (the root vCPU's
    /// runtime shares the same `Arc`, held by the `CompiledModule`). `None` for fiber-free modules.
    /// `std::sync::Arc`/`Mutex<Option<…>>` like `env` (set once on the setup thread, read at spawns).
    fiber_table: Mutex<Option<std::sync::Arc<SharedFiberTable>>>,
    threads: Mutex<Threads>,
    /// **Durable** (slice 3.3): `thread.spawn` requests *deferred* during a freeze. While the window
    /// state ≠ NORMAL the run is single-worker, so a spawn does not start an OS thread; it records the
    /// child here (returning its handle) and the child runs **inline after the root unwinds**
    /// ([`Domain::drive_frozen_spawns`]) — mirroring the interpreter, which enqueues a child and
    /// dispatches it only once the spawning vCPU yields. Empty on a non-durable run.
    pending_spawns: Mutex<Vec<PendingSpawn>>,
    /// **Durable** (slice 3.3): residue of the deferred children that unwound under the freeze, pushed
    /// by [`Domain::drive_frozen_spawns`]; `run_inner` drains it into the run's residue so a snapshot
    /// records the children (a thaw re-attaches them). Empty on a non-durable run.
    frozen_vcpus: Mutex<Vec<crate::FrozenVCpu>>,
    /// §12.8 4A.5 follow-up A: concurrent durable children that **finished** (didn't freeze) during a
    /// run, recorded by [`run_child`]. On a freeze the coordinator turns *every* one into a
    /// `completed_result` residue (so the spawner's `thread.join` of a child that finished before the
    /// freeze point resolves on thaw — the host-side Done cell isn't in the snapshot). Emitting them all
    /// keeps the thaw's per-parent join table dense so every handle still resolves. Drained on a freeze;
    /// discarded on a non-freeze run.
    completed_children: Mutex<Vec<CompletedChild>>,
    /// **Durable** (slice 3.4): residue of the fibers a *spawned child* flattened with its own
    /// `freeze_drive` (a child that owns `cont.*` fibers). Accumulated by [`Domain::run_child_inline`];
    /// `run_inner` merges it into the run's `frozen_out` (the root's own fibers are flattened directly
    /// there). Empty on a non-durable run or a thread-only domain.
    frozen_fibers: Mutex<Vec<crate::FrozenFiber>>,
    /// **Durable single-worker** (slice 3.4: nested spawns): per-vCPU join tables, keyed by the
    /// **spawning** task. A nested child's guest handle is its index *within its parent's* table —
    /// matching the interpreter's per-vCPU `threads`, so the handle the guest spills is byte-identical
    /// across backends. Populated only on the durable single-worker path (freeze defer / thaw
    /// re-attach); empty (and unused) on the global OS-thread path, which keeps using [`Threads::cells`].
    dchildren: Mutex<HashMap<u64, DTable>>,
    /// The inline vCPU currently executing on the single worker (routing for `defer_spawn` +
    /// `thread_join`); the root is `0`. Set around each inline child run; `0` while the root runs.
    cur_task: Mutex<u64>,
    /// Monotonic task-id allocator for the durable single-worker path (root = `0`, first spawn = `1`),
    /// matching the interpreter's `next_task` so `FrozenVCpu.task` is byte-identical.
    next_task: Mutex<u64>,
    futex: Mutex<HashMap<u64, FutexEntry>>,
    futex_cv: Condvar,
    /// §15 spawn quota: max **concurrently-live** vCPUs (incl. the root) this domain may have, clamped
    /// to [`MAX_VCPUS`]. Exceeding it is a clean `ThreadFault`. Bounds `Threads::live` (concurrent),
    /// matching the interpreter's `s.live` — a spawn-join loop is fine (a finished vCPU frees its slot).
    max_vcpus: usize,
    /// §5 W3 Stage 3: a trap-time backtrace capture handed up by a **spawned** vCPU when it trapped.
    /// The per-thread capture lives in `trap_shim.c`'s thread-local (filled by the SIGSEGV/SIGBUS
    /// handler or the explicit-trap helper) and would be lost when the worker thread ends, so the
    /// dying worker publishes it here for the run thread to symbolize after `join_all`. `(pc, return
    /// addresses)`; last-wins (matching the last-wins trap cell — any trapping frame's chain is a
    /// valid kill backtrace). `None` until a spawned vCPU traps. Host-side observability (§2a).
    trap_capture: Mutex<Option<(usize, Vec<usize>, i64)>>,
    /// **Concurrent durable STW** (Phase-4 Slice A, 4A.4): the quiesce barrier for a multi-worker
    /// async freeze. `quiesce` holds the count of vCPUs still to reach a quiescent (unwound) state;
    /// the coordinator waits on `quiesce_cv` until it hits 0, then runs the existing single-worker
    /// freeze-drive. The *same* `quiesce` lock guards the active shadow-SP scratch during each worker's
    /// unwind, so that single shared word has one owner at a time (the R10 seam). Engaged only when
    /// `concurrent_durable` is set — by the 4A.5 concurrent multi-vCPU entry alone. On every existing
    /// path (single-worker freeze defer, thaw, ordinary runs) the lock is never taken and the flag is
    /// false, so behavior is byte-identical. See [`Domain::quiesce_arrive`] / [`Domain::quiesce_wait_all`].
    quiesce: Mutex<usize>,
    quiesce_cv: Condvar,
    concurrent_durable: AtomicBool,
    /// §12.8 concurrent-thaw stage 3: count of vCPUs currently **blocked** (parked in `atomic.wait` or
    /// `thread.join`). A `notify`/`join`-completer must be a *live, not-parked* vCPU, so an infinite wait
    /// can never be satisfied once every live vCPU is parked (`live == parked`): `futex_wait`'s
    /// `peers_live` (`live > parked`) then fails it closed (`ThreadFault`) instead of hanging — catching a
    /// **mutual** wait/join deadlock (two vCPUs each blocked on the other), not just a lone waiter.
    parked: AtomicUsize,
}

/// One vCPU's join table on the durable single-worker path (slice 3.4): its spawned children's
/// completion cells by per-vCPU handle (the index), plus the per-handle "already joined" flag. The
/// handle namespace is per-spawning-vCPU, mirroring the interpreter's per-vCPU `threads`.
#[derive(Default)]
struct DTable {
    cells: Vec<std::sync::Arc<Done>>,
    joined: Vec<bool>,
}

#[derive(Default)]
struct Threads {
    /// §15 **concurrently-live** vCPUs — the root (1) plus every spawned vCPU that hasn't finished.
    /// Incremented under this lock at a successful `thread.spawn`, decremented when a spawned vCPU's
    /// computation ends ([`run_child`]). The §15 quota bounds *this* (concurrent liveness, like the
    /// interpreter's `s.live`), **not** `cells.len()` (the cumulative handle table that never shrinks —
    /// symmetric with the interpreter's per-vCPU `threads` Vec). Starts at 1 (the root) via
    /// [`Domain::new`]; `Default` is 0 only for the unused loom path.
    live: usize,
    /// Completion cells, indexed by thread handle (a masked, generation-free table index, §3c).
    cells: Vec<std::sync::Arc<Done>>,
    /// Per-handle "already joined" flag — a second `thread.join` of the same vCPU is inert (traps),
    /// matching the interpreter oracle and the IR contract.
    joined: Vec<bool>,
    /// OS-thread handles, joined by `run_inner` at run end so no vCPU outlives the window/code.
    joins: Vec<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct FutexEntry {
    /// Bumped by `notify` so parked waiters re-check and observe a wake (vs a spurious one).
    generation: u64,
    waiters: u32,
}

impl Domain {
    pub(crate) fn new(max_vcpus: usize) -> Domain {
        Domain {
            env: Mutex::new(None),
            fiber_table: Mutex::new(None),
            // `live` starts at 1: the root vCPU (the main thread running the entry) counts toward the
            // §15 quota, like the interpreter's `s.live`.
            threads: Mutex::new(Threads {
                live: 1,
                ..Threads::default()
            }),
            pending_spawns: Mutex::new(Vec::new()),
            frozen_vcpus: Mutex::new(Vec::new()),
            completed_children: Mutex::new(Vec::new()),
            frozen_fibers: Mutex::new(Vec::new()),
            dchildren: Mutex::new(HashMap::new()),
            cur_task: Mutex::new(0),
            next_task: Mutex::new(1), // root is task 0; first spawn is 1
            futex: Mutex::new(HashMap::new()),
            futex_cv: Condvar::new(),
            max_vcpus: max_vcpus.clamp(1, MAX_VCPUS),
            trap_capture: Mutex::new(None),
            quiesce: Mutex::new(0),
            quiesce_cv: Condvar::new(),
            concurrent_durable: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
        }
    }

    /// Publish a **spawned** vCPU's raw trap-time backtrace capture (§5 W3 Stage 3), last-wins. Called
    /// from a dying worker ([`run_child`]) when it trapped, so the run thread can symbolize a trap that
    /// originated off the run thread (whose own thread-local capture it can't see).
    pub(crate) fn publish_trap_capture(&self, cap: (usize, Vec<usize>, i64)) {
        *lock(&self.trap_capture) = Some(cap);
    }

    /// Take the trap-time backtrace capture a spawned vCPU published (§5 W3 Stage 3), if any — `(pc,
    /// return-address chain, trapping fiber handle)`. Called on the run thread after
    /// [`Self::join_all`], so the publishing worker has finished.
    pub(crate) fn take_trap_capture(&self) -> Option<(usize, Vec<usize>, i64)> {
        lock(&self.trap_capture).take()
    }

    /// Supply the per-run [`Env`] once the call-trampoline / window addresses are known (post-finalize,
    /// before the run). Called only on the setup thread before any vCPU spawns.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_env(
        &self,
        mem_base: u64,
        fn_table_base: u64,
        trap_out: *mut i64,
        call_tramp: FiberCallTramp,
        fault: (usize, usize),
        fiber_cfg: Option<(u32, u64)>,
        fiber_table: Option<std::sync::Arc<SharedFiberTable>>,
        epoch_addr: usize,
        durable: bool,
    ) {
        *lock(&self.env) = Some(Env {
            mem_base,
            fn_table_base,
            trap_out,
            call_tramp,
            fault_lo: fault.0,
            fault_hi: fault.1,
            fiber_cfg,
            epoch_addr,
            durable,
        });
        *lock(&self.fiber_table) = fiber_table;
    }

    /// The domain-shared fiber table (for a spawned vCPU to build its `FiberRuntime` over).
    fn fiber_table(&self) -> Option<std::sync::Arc<SharedFiberTable>> {
        lock(&self.fiber_table).clone()
    }

    fn env(&self) -> Env {
        lock(&self.env).expect("Domain::set_env before any thread op")
    }

    /// **Concurrent durable STW** (Phase-4 Slice A, 4A.4). Arm the quiesce barrier for `runners`
    /// concurrently-unwinding vCPUs.
    ///
    /// NOTE: this barrier ([`Self::arm_quiesce`] / [`Self::quiesce_arrive`] / [`Self::quiesce_wait_all`])
    /// is **not on the live path**. Stage (ii) chose `run_inner`'s existing `join_all` as the
    /// coordinator-wait: with per-context shadow-SP (stage i), each child's freeze-unwind into its own
    /// region simply completes its OS thread, and `join_all` blocks until all have — so an explicit
    /// barrier is unnecessary. It is retained as a **loom-verified primitive** (model:
    /// `loom_quiesce_barrier_never_hangs_with_per_context_sp`) in case a future park-in-place quiesce
    /// (workers that stop *without* ending their OS thread) needs it. Exercised only under `cfg(loom)`.
    #[cfg_attr(not(loom), allow(dead_code))]
    pub(crate) fn arm_quiesce(&self, runners: usize) {
        *lock(&self.quiesce) = runners;
        self.concurrent_durable.store(true, Ordering::Release);
    }

    /// True on a concurrent multi-worker durable run (4A.5). `thread_spawn` reserves a per-context
    /// shadow context for a child spawned during NORMAL (so a later freeze makes it self-unwind into its
    /// own region); `run_child` seeds the durable shadow-base register and records freeze residue. Off
    /// on every existing path, so they are byte-identical.
    pub(crate) fn is_concurrent_durable(&self) -> bool {
        self.concurrent_durable.load(Ordering::Acquire)
    }

    /// Engage the concurrent durable path (§12.8 4A.5 stage ii) for a freezable multi-vCPU run. Set once
    /// at run setup, before any child is spawned. The coordinator joins concurrent children via the
    /// existing `join_all` (each child's freeze-unwind into its own per-context region completes its OS
    /// thread); no shared active-SP word and no quiesce lock — the per-context relocation (stage i)
    /// made that serialization unnecessary.
    pub(crate) fn engage_concurrent_durable(&self) {
        self.concurrent_durable.store(true, Ordering::Release);
    }

    /// **Concurrent durable STW** (Phase-4 Slice A) — a worker reaches the quiesce barrier. §12.8 4A.5:
    /// each context unwinds into its **own** region's shadow-SP word (per-context, no shared scratch
    /// since the relocation), so `unwind` runs **concurrently, outside the lock** — workers no longer
    /// serialize on a single active-SP word. The lock guards only the *join*: decrement the runner
    /// count and, when the last worker arrives, wake the coordinator. The notify can't be lost: both
    /// the decrement-to-zero and the coordinator's wait hold `quiesce`, so the coordinator either sees
    /// `runners == 0` before parking or is woken after (O-A4.4 — the same invariant as the futex).
    #[cfg_attr(not(loom), allow(dead_code))]
    pub(crate) fn quiesce_arrive(&self, unwind: impl FnOnce()) {
        unwind(); // concurrent: each context spills into its own per-context region word (4A.5)
        let mut runners = lock(&self.quiesce);
        *runners -= 1;
        if *runners == 0 {
            self.quiesce_cv.notify_all();
        }
    }

    /// **Concurrent durable STW** (Phase-4 Slice A, 4A.4) — the coordinator waits until every worker
    /// has quiesced (unwound to base), then it alone runs the existing single-worker freeze-drive.
    /// Never hangs (O-A4.4): the wait re-checks `runners` under `quiesce`.
    #[cfg_attr(not(loom), allow(dead_code))]
    pub(crate) fn quiesce_wait_all(&self) {
        let mut runners = lock(&self.quiesce);
        while *runners > 0 {
            runners = self
                .quiesce_cv
                .wait(runners)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Drain the durable freeze residue collected by inline children (slice 3.3) — called by
    /// `run_inner` after the root unwinds, so the durable entry can hand it to the embedder for the
    /// snapshot. Empty on a non-durable / non-freeze run.
    pub(crate) fn take_frozen_vcpus(&self) -> Vec<crate::FrozenVCpu> {
        std::mem::take(&mut lock(&self.frozen_vcpus))
    }

    /// §12.8 4A.5 follow-up A: drain the **completed** concurrent children as `completed_result`
    /// residue (every one, so the thaw's per-parent join table stays dense and all handles resolve).
    /// Each thaws as a no-re-run cell pre-filled with the recorded `thread.join` result. Called by the
    /// coordinator on a freeze, after `join_all`.
    pub(crate) fn take_completed_children_residue(&self) -> Vec<crate::FrozenVCpu> {
        std::mem::take(&mut *lock(&self.completed_children))
            .into_iter()
            .map(|c| crate::FrozenVCpu {
                task: c.task as usize,
                parent_task: c.parent as usize,
                func: c.func,
                args: c.args,
                shadow_sp: 0, // inert — a completed child is not re-run
                completed_result: Some(c.result),
            })
            .collect()
    }

    /// Drain the child-owned fiber residue (slice 3.4) — the fibers spawned children flattened with
    /// their own `freeze_drive`. `run_inner` merges this into the run's `frozen_out`.
    pub(crate) fn take_frozen_fibers(&self) -> Vec<crate::FrozenFiber> {
        std::mem::take(&mut lock(&self.frozen_fibers))
    }

    /// Join every spawned OS thread (run teardown). After this returns no vCPU is still touching the
    /// window or executable code, so `run_inner` can free them.
    pub(crate) fn join_all(&self) {
        // Drain handles out of the lock first (joining holds no lock; a joining child may still touch
        // `threads` via nested ops, though by teardown all guest code has returned).
        let joins: Vec<_> = std::mem::take(&mut lock(&self.threads).joins);
        // §12.8 concurrent-thaw stage 3: the thread calling `join_all` has finished its own guest code
        // (it's tearing down), so it can never `notify` a still-parked vCPU. Count it as blocked while it
        // joins — else a vCPU left parked in a genuine deadlock (e.g. a sibling that unwound, propagating
        // a trap, before the parked one's own deadlock check fired) would see this joiner as a live,
        // not-parked "potential notifier" (`live > parked`) and wait forever, hanging `join_all` too.
        let _pg = ParkGuard::new(&self.parked);
        for j in joins {
            let _ = j.join();
        }
    }
}

/// `MutexGuard` helper that recovers from poisoning (a panicking holder means the domain is already
/// being killed; we still want the data). loom's `Mutex` has no poisoning, so it returns the guard.
#[cfg(not(loom))]
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
#[cfg(loom)]
fn lock<T>(m: &Mutex<T>) -> loom::sync::MutexGuard<'_, T> {
    m.lock().unwrap()
}

/// In/out cell smuggled through the `Entry`-shaped [`child_entry`] so the guarded runner can call the
/// guest entry (and a fault can `longjmp` past it).
struct ChildCall {
    env: Env,
    code: u64,
    sp: u64,
    arg: u64,
    ret: i64,
}

/// `Entry`-shaped shim (so it runs under [`mem::run_guarded_range`] with no new C shim): call the guest
/// thread entry `code(sp, arg)` via the shared call-trampoline, stash the result.
extern "C" fn child_entry(
    a: *const i64,
    _r: *mut i64,
    _m: *mut u8,
    _t: *const core::ffi::c_void,
    _tc: *mut i64,
) {
    // SAFETY: `a` is the `&mut ChildCall` the runner passed; all its fields are live for the call.
    unsafe {
        let c = a as *mut ChildCall;
        let env = (*c).env;
        let v = (env.call_tramp)(
            (*c).code,
            env.mem_base,
            env.fn_table_base,
            env.trap_out as u64,
            (*c).sp,
            (*c).arg,
        );
        (*c).ret = v as i64;
    }
}

/// What a spawned OS thread runs: set up its own fiber runtime (for `cont.*`), arm the guard, call the
/// guest entry, then publish the result/trap into its completion cell.
/// §12.8 4A.5 stage (ii): a **concurrent durable** child's per-context durability state. Present only
/// when a freezable (interruptible) durable run spawns a child as a real OS thread during NORMAL — the
/// child reserves its own top-down shadow context up front so it can self-unwind into it if a freeze
/// fires mid-run. `None` on non-durable / single-worker / deferred spawns.
struct DurableChild {
    /// Global task id (monotonic spawn order; matches the interp + the deferred path), recorded in the
    /// child's `FrozenVCpu` residue.
    task: u64,
    /// The spawning task (`parent_task`) — the root (`0`) for a flat spawn.
    parent: u64,
    /// The reserved top-down shadow context the child unwinds into (kept on a freeze-unwind so a thaw
    /// re-spawns it there; freed on a genuine finish).
    ctx: usize,
    /// The child's entry function index — recorded in its residue so a thaw re-spawns it.
    func_idx: u32,
    /// §12.8 concurrent-thaw stage 2: `None` on a fresh spawn (the child starts `NORMAL` at its entry,
    /// region seeded to the empty frame base). `Some(extent)` on a **thaw** re-spawn: the child starts
    /// `REWINDING` from its restored shadow extent (its region's SP word set to `extent`, its per-context
    /// thaw word set `REWINDING`), so it rewinds its frozen frames concurrently with its siblings + root.
    thaw_extent: Option<u64>,
}

/// §12.8 4A.5 follow-up A: a concurrent durable child that **finished** (didn't freeze). Recorded so a
/// freeze can carry its `thread.join` result in the artifact (`FrozenVCpu::completed_result`), to be
/// delivered on thaw without re-running the child.
struct CompletedChild {
    task: u64,
    parent: u64,
    func: i32,
    args: Vec<i64>,
    result: i64,
}

struct SpawnArgs {
    env: Env,
    code: u64,
    sp: u64,
    arg: u64,
    /// §12 dense vCPU id seeded into this child's per-vCPU TLS register (`vcpu.tls`). Root is 0, so a
    /// spawned vCPU takes `handle + 1` (handles are 0-based, cumulative spawn order).
    vcpu_id: i64,
    /// §12.8 4A.5 stage (ii): present on a concurrent durable child (see [`DurableChild`]).
    durable_child: Option<DurableChild>,
    done: std::sync::Arc<Done>,
    /// The owning [`Domain`] — so this vCPU can drop its §15 concurrent-live count when it finishes.
    /// The domain outlives every spawned thread (`run_inner` joins them at run end), so the pointer
    /// stays valid for the thread's lifetime.
    dom: *const Domain,
}
// SAFETY: same contract as `Env` — the raw pointers are the run's shared window/trap cell, and a fresh
// OS thread is the sole user of its `SpawnArgs` until it stores into the (synchronized) `Done` cell.
unsafe impl Send for SpawnArgs {}

/// A `thread.spawn` **deferred** during a durable freeze (slice 3.3): the child is recorded here at
/// the spawn (its handle already returned to the guest) and run inline, in spawn order, once the root
/// has unwound ([`Domain::drive_frozen_spawns`]). Carries everything the inline run needs.
struct PendingSpawn {
    /// This child's **global** task id (monotonic spawn order; matches the interp), recorded in its
    /// `FrozenVCpu` and used as `cur_task` while it runs inline (so a grandchild it spawns is attributed
    /// to it).
    task: u64,
    /// The task that spawned it (its `parent_task`) — the root (`0`) or another child, for nested spawns.
    parent: u64,
    /// The reserved top-down durable shadow context (kept on a freeze-unwind, freed on a genuine
    /// finish) the child runs in.
    ctx: usize,
    code: u64,
    func_idx: u32,
    sp: u64,
    arg: u64,
    done: std::sync::Arc<Done>,
}

std::thread_local! {
    /// §12.8 4A.5 follow-up B.2: the **spawning-task source** for a concurrent durable run. `Some(t)` on
    /// an OS thread running concurrent durable child vCPU `t` ([`run_child`] seeds it); `None` on the
    /// root's own thread (which never runs `run_child`). [`thread_spawn`] reads it to attribute a
    /// (possibly nested) concurrent spawn's `parent_task` to the **spawning** vCPU — a per-OS-thread read,
    /// so concurrent spawners never race (unlike the shared `cur_task`, which only the single-worker
    /// inline/thaw paths maintain). `None` ⇒ the root (task `0`).
    static CONCURRENT_SPAWN_TASK: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

fn run_child(a: SpawnArgs) {
    let env = a.env;
    // §12 seed this vCPU's per-vCPU TLS register to its dense id before any guest code runs.
    crate::vcpu_tls::seed(a.vcpu_id);
    // §12.8 4A.5 stage (ii): a concurrent durable child points its durable shadow-base register at its
    // own region AND initialises that region's shadow-SP word to the empty frame base, so if a freeze
    // fires its instrumented code spills into *its* per-context region (concurrent with siblings, no
    // shared word). Without the init the word would read 0 and the unwind would spill over the reserve
    // header — corrupting the state word. SAFETY: durable run ⇒ the reserve is committed RW.
    if let Some(dc) = &a.durable_child {
        let region = fiber_rt::shadow_region_base(dc.ctx);
        crate::durable_shadow::seed(region);
        match dc.thaw_extent {
            // §12.8 concurrent-thaw stage 2: a **thaw** re-spawn rewinds from its restored extent — set
            // its region's SP word to that extent and its own per-context thaw word `REWINDING`, so the
            // prologue dispatches into the rewind (concurrent with siblings, no shared word).
            Some(extent) => unsafe {
                fiber_rt::write_shadow_sp(env.mem_base, region, extent);
                fiber_rt::window_set_rewinding(env.mem_base, dc.ctx);
            },
            // A fresh spawn starts NORMAL at its entry, region empty (frame base).
            None => unsafe {
                fiber_rt::write_shadow_sp(env.mem_base, region, fiber_rt::shadow_frame_base(dc.ctx))
            },
        };
        // §12.8 4A.5 follow-up B.2: record this OS thread's spawning task, so a *nested* `thread.spawn`
        // it makes attributes the grandchild's `parent_task` to **this** child (read per-OS-thread in
        // `thread_spawn`). The OS thread is fresh per child, so the value is naturally scoped to it.
        CONCURRENT_SPAWN_TASK.with(|c| c.set(Some(dc.task)));
    }
    // Arm this OS thread's detect-and-kill recovery (idempotent; handler is process-wide, recovery is
    // thread-local — §5 / `mem::install_guard`).
    mem::install_guard();
    // A vCPU that uses `cont.*` gets its own fiber *execution context* over the **domain-shared**
    // fiber table (D57 3b-ii: one handle namespace + quota; the table outlives every vCPU — it is
    // held by the `CompiledModule` and the `Domain`, both joined-after). SAFETY: `a.dom` is the
    // run's live `Domain` (joined at run end).
    let mut frt = env.fiber_cfg.map(|(tid, mask)| {
        let table = unsafe { (*a.dom).fiber_table() }
            .expect("fiber_cfg set ⇒ the domain fiber table is set");
        let mut rt = FiberRuntime::new(table, tid, mask);
        rt.set_call_tramp(env.call_tramp);
        // §12.8 4A.5 follow-up B: a **concurrent durable** child that owns fibers needs `mem_base` +
        // the `durable` flag on its own runtime, so its `cont.resume` swap repoints the active
        // shadow-SP at the fiber's region and the freeze driver records flattened fibers as residue
        // (mirrors `run_child_inline`). Off the durable concurrent path this is inert.
        if a.durable_child.is_some() {
            rt.set_durable_env(env.mem_base, env.durable);
        }
        Box::new(rt)
    });
    let prev = frt
        .as_mut()
        .map(|b| fiber_rt::set_current(&mut **b as *mut FiberRuntime));

    let mut call = ChildCall {
        env,
        code: a.code,
        sp: a.sp,
        arg: a.arg,
        ret: 0,
    };
    // SAFETY: `child_entry` honours the `Entry` ABI; `call` outlives the run; a guest fault in the
    // window range unwinds back here (this vCPU's stack is abandoned — the domain is being killed).
    let faulted = unsafe {
        mem::run_guarded_range(
            child_entry as *const () as *const u8,
            &mut call as *mut ChildCall as *const i64,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            env.fault_lo,
            env.fault_hi,
        )
    };
    // §12.8 4A.5 follow-up B: a concurrent durable child that **owns fibers** must flatten the ones it
    // parked into their shadow regions — its own `freeze_drive`, the concurrent mirror of
    // `run_child_inline`'s (the root's drive ran before this child existed). Run it while the child's
    // runtime is still `CURRENT_RT` and the window is `UNWINDING`; drain the child's fiber residue into
    // the domain accumulator (`run_inner` merges it; canonical sort-by-slot keeps the artifact
    // order-independent). SAFETY: durable run ⇒ committed reserve; live `Domain`.
    if a.durable_child.is_some()
        && !faulted
        && unsafe { fiber_rt::window_is_unwinding(env.mem_base) }
    {
        if let Some(b) = frt.as_mut() {
            let rt = &mut **b as *mut FiberRuntime;
            unsafe {
                fiber_rt::freeze_drive(rt, env.trap_out as u64);
                let child_frozen = fiber_rt::take_frozen(rt);
                if !child_frozen.is_empty() {
                    lock(&(*a.dom).frozen_fibers).extend(child_frozen);
                }
            }
        }
    }
    if let Some(p) = prev {
        fiber_rt::set_current(p);
    }

    let (result, trap) = if faulted {
        // SAFETY: `trap_out` is the run's live trap cell (an `AtomicI64`'s storage).
        unsafe { store_trap(env.trap_out, TrapKind::MemoryFault as i64) };
        (0, TrapKind::MemoryFault as i64)
    } else {
        // A non-memory trap (DivByZero, ThreadFault, …) set the shared cell from inside the run.
        // SAFETY: live trap cell.
        let t = unsafe { load_trap(env.trap_out) };
        (call.ret, t)
    };
    // §12.8 4A.5 stage (ii): if this concurrent durable child observed a freeze, its instrumented code
    // has now unwound into *its own* region; record its `FrozenVCpu` residue (extent = its region's
    // shadow-SP word) so the coordinator's snapshot (taken after `join_all`) captures its continuation,
    // and keep its context for the thaw to re-spawn it there. A genuine finish (no freeze) frees the
    // context for reuse. SAFETY: `a.dom` is the run's live `Domain` (joined at run end).
    if let Some(dc) = &a.durable_child {
        let region = fiber_rt::shadow_region_base(dc.ctx);
        let extent = if faulted {
            0
        } else {
            unsafe { fiber_rt::read_shadow_sp(env.mem_base, region) }
        };
        // A freeze-unwind iff the window is UNWINDING **and** the child actually spilled past its frame
        // base — a child that ran to a genuine finish under an UNWINDING window left its region empty.
        let froze = !faulted
            && unsafe { fiber_rt::window_is_unwinding(env.mem_base) }
            && extent > fiber_rt::shadow_frame_base(dc.ctx);
        if froze {
            unsafe {
                lock(&(*a.dom).frozen_vcpus).push(crate::FrozenVCpu {
                    task: dc.task as usize,
                    parent_task: dc.parent as usize,
                    func: dc.func_idx as i32,
                    args: vec![a.sp as i64, a.arg as i64],
                    shadow_sp: extent,
                    completed_result: None, // a spilled (frozen) child re-runs on thaw
                });
            }
        } else {
            // A genuine finish: free the context for reuse, and record the child's result so a freeze
            // (if one is in flight / about to be) can carry it as residue — a `thread.join` of a child
            // that finished before the freeze point must resolve on thaw. Discarded if the run doesn't
            // freeze. SAFETY: live `Domain`.
            unsafe {
                if let Some(table) = (*a.dom).fiber_table() {
                    table.free_vcpu_context(dc.ctx);
                }
                if trap == 0 {
                    lock(&(*a.dom).completed_children).push(CompletedChild {
                        task: dc.task,
                        parent: dc.parent,
                        func: dc.func_idx as i32,
                        args: vec![a.sp as i64, a.arg as i64],
                        result,
                    });
                }
            }
        }
    }
    // §5 W3 Stage 3: if this spawned vCPU trapped, hand its trap-time backtrace capture (the SIGSEGV
    // handler's, or the explicit-trap helper's — both in this thread's `trap_shim.c` thread-local) to
    // the domain before this worker ends, else it would be lost and the run thread couldn't symbolize
    // a trap that originated here. SAFETY: `a.dom` is the run's live `Domain` (joined at run end).
    if trap != 0 {
        if let Some(cap) = mem::take_trap_frame() {
            unsafe { (*a.dom).publish_trap_capture(cap) };
        }
    }
    // §15: this vCPU's computation has ended — free its concurrent-live slot *before* publishing the
    // result, so a `thread.join` that then observes completion already sees the quota slot freed (a
    // spawn-join loop can't transiently false-trap). The domain outlives all spawned threads, so the
    // pointer is live. SAFETY: `a.dom` is the run's `Domain` (joined at run end).
    unsafe {
        let mut t = lock(&(*a.dom).threads);
        t.live -= 1;
    }
    let mut st = lock(&a.done.state);
    *st = Some((result, trap));
    a.done.cv.notify_all();
}

/// The running vCPU's env, found via the baked `Domain` pointer the thunks receive.
///
/// `thread.spawn` thunk: launch a new vCPU OS thread running `funcs[func_idx](sp, arg)`; return its
/// `i32` handle. Traps (`ThreadFault`, returns `-1`) on a thread-bomb.
///
/// # Safety
/// `sched` is the run's live `Domain`; the other args are the threaded context (window / table /
/// trap cell). Compiled only for the real (non-loom) executor.
pub(crate) unsafe extern "C" fn thread_spawn(
    sched: *const Domain,
    _mem_base: u64,
    _fn_table_base: u64,
    trap_out: u64,
    func_idx: u32,
    sp: u64,
    arg: u64,
) -> i32 {
    let dom = &*sched;
    let env = dom.env();
    let entry = (env.fn_table_base as *const FnEntry).add(func_idx as usize);
    let code = (*entry).code();
    // Durable freeze/thaw in progress (slice 3.3): the run is **single-worker** while the window
    // state ≠ NORMAL (mirroring the interp's `workers=1`), so don't start an OS thread — *defer* the
    // child (record it, return its handle) and run it inline after the spawning vCPU yields
    // ([`Domain::drive_frozen_spawns`]), exactly as the interp enqueues a child and dispatches it only
    // once the spawning vCPU unwinds. This keeps the one shared set of durable control words unraced
    // *and* reproduces the interp's side-effect interleaving (root runs to its unwind point first).
    if env.durable && fiber_rt::window_is_durable_active(env.mem_base) {
        return defer_spawn(dom, code, func_idx, sp, arg, trap_out);
    }
    // §12.8 4A.5 stage (ii): on a freezable (interruptible) durable run, a child spawned during NORMAL
    // runs as a real OS thread but reserves its own top-down shadow context now, so a later freeze can
    // make it self-unwind into its own per-context SP word (concurrent, lock-free — stage i). Its global
    // task id matches the interp/deferred path (seeded into `vcpu.tls`). `None` on every existing path,
    // so behavior is unchanged there.
    let durable_child = if env.durable && dom.is_concurrent_durable() {
        // §12.8 4A.5 follow-up B.2: a **concurrent** child that spawns a grandchild attributes the
        // grandchild's `parent_task` to itself via the per-OS-thread spawning-task source (`run_child`
        // seeded it) — not the shared `cur_task`, which the concurrent path never maintains and which
        // would race across spawners. The root's own thread leaves it `None` ⇒ task `0`.
        let table = match dom.fiber_table() {
            Some(t) => t,
            None => {
                store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                return -1;
            }
        };
        let ctx = match table.reserve_vcpu_context() {
            Some(c) => c,
            None => {
                store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                return -1;
            }
        };
        let parent = CONCURRENT_SPAWN_TASK.with(|c| c.get()).unwrap_or(0);
        let task = {
            let mut nt = lock(&dom.next_task);
            let t = *nt;
            *nt += 1;
            t
        };
        Some(DurableChild {
            task,
            parent,
            ctx,
            func_idx,
            thaw_extent: None, // a fresh spawn starts NORMAL; only a thaw re-spawn rewinds
        })
    } else {
        None
    };
    let done = std::sync::Arc::new(Done {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    let handle = {
        let mut t = lock(&dom.threads);
        // §15: bound *concurrent* live vCPUs (root + unfinished spawns), not the cumulative handle
        // table — so a spawn-join loop never trips, matching the interpreter.
        if t.live >= dom.max_vcpus {
            if let Some(dc) = &durable_child {
                if let Some(table) = dom.fiber_table() {
                    table.free_vcpu_context(dc.ctx);
                }
            }
            store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
            return -1;
        }
        let idx = t.cells.len();
        t.cells.push(std::sync::Arc::clone(&done));
        t.joined.push(false);
        let vcpu_id = durable_child
            .as_ref()
            .map(|d| d.task as i64)
            .unwrap_or(idx as i64 + 1); // dense id seed (root 0; children 1, 2, …)
        let args = SpawnArgs {
            env,
            code,
            sp,
            arg,
            vcpu_id,
            durable_child,
            done,
            dom: sched,
        };
        let jh = std::thread::Builder::new()
            .name(format!("svm-vcpu-{idx}"))
            .spawn(move || run_child(args));
        match jh {
            Ok(jh) => {
                t.joins.push(jh);
                // Count the new vCPU as live *before* releasing the lock, so its own completion
                // decrement (which takes this lock) can't underflow.
                t.live += 1;
                idx as i32
            }
            Err(_) => {
                // Out of OS threads: pop the cell we reserved and trap (no `live` change). The child's
                // reserved shadow context (if any) leaks for this run — a spawn failure aborts the run.
                t.cells.pop();
                t.joined.pop();
                store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                -1
            }
        }
    };
    handle
}

/// **Deferred single-worker** `thread.spawn` (slice 3.3): while a durable freeze/thaw is in progress
/// (window state ≠ NORMAL) the run is single-worker, so this does **not** start an OS thread — it
/// reserves the child's shadow context + completion cell, records the request, and returns the handle.
/// The child runs **inline after the spawning vCPU unwinds** ([`Domain::drive_frozen_spawns`]),
/// mirroring the interpreter, which enqueues a child and dispatches it only once the spawning vCPU
/// yields — so the side-effect interleaving (and the frozen window) is byte-identical across backends.
///
/// # Safety
/// As [`thread_spawn`]: `dom` is the run's live `Domain`; `code` is a guest entry trampoline;
/// `trap_out` is the live trap cell; the durable reserve `[0, DURABLE_RESERVE)` is committed RW.
unsafe fn defer_spawn(
    dom: &Domain,
    code: u64,
    func_idx: u32,
    sp: u64,
    arg: u64,
    trap_out: u64,
) -> i32 {
    // Reserve this child's shadow context top-down (`MAX_SHADOW_CTX`, −1, …) so it can't collide with
    // a fiber's `slot+1` region; fail closed (`ThreadFault`) if the reserve is full — the vCPU pool
    // growing down would meet the fiber pool growing up (same as the interp). A durable run always has
    // the table (`uses_fibers || uses_threads`).
    let table = match dom.fiber_table() {
        Some(t) => t,
        None => {
            store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
            return -1;
        }
    };
    let ctx = match table.reserve_vcpu_context() {
        Some(c) => c,
        None => {
            store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
            return -1;
        }
    };
    let done = std::sync::Arc::new(Done {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    // §15 concurrent-live quota (the global counter, like the OS-thread path) — bound it the same way.
    {
        let mut t = lock(&dom.threads);
        if t.live >= dom.max_vcpus {
            table.free_vcpu_context(ctx);
            store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
            return -1;
        }
        t.live += 1;
    }
    // The **spawning** vCPU (cur_task) and a fresh **global** task id for the child (monotonic, matching
    // the interp). The guest handle is the child's index *within the spawning vCPU's* table (per-vCPU
    // namespace) — so a nested grandchild's handle is `0` in its parent's table, byte-identical to the
    // interp, not a global running index.
    let parent = *lock(&dom.cur_task);
    let task = {
        let mut nt = lock(&dom.next_task);
        let t = *nt;
        *nt += 1;
        t
    };
    let handle = {
        let mut dc = lock(&dom.dchildren);
        let tbl = dc.entry(parent).or_default();
        let h = tbl.cells.len();
        tbl.cells.push(std::sync::Arc::clone(&done));
        tbl.joined.push(false);
        h
    };
    lock(&dom.pending_spawns).push(PendingSpawn {
        task,
        parent,
        ctx,
        code,
        func_idx,
        sp,
        arg,
        done,
    });
    handle as i32
}

impl Domain {
    /// Run one child's guest entry **inline** on this thread (single-worker, slice 3.3): set up its
    /// fiber execution context (for `cont.*`), arm a **nested** detect-and-kill recovery
    /// (`run_guarded_range` saves/restores the parent run's recovery state itself, like `invoke_extra`),
    /// call `code(sp, arg)` via the shared call-trampoline, and return `(result, trap, faulted)`. The
    /// caller is responsible for the durable control words (state + active shadow-SP) around the call.
    ///
    /// # Safety
    /// `env` is the live run's env; `code` is a guest entry trampoline; called with no other guest code
    /// on this thread (the root returned / is parked in a join), the durable reserve committed RW.
    unsafe fn run_child_inline(
        &self,
        env: Env,
        code: u64,
        sp: u64,
        arg: u64,
        vcpu_id: i64,
    ) -> (i64, i64, bool) {
        // §12 this inline child runs on *this* (the root's) OS thread, so seed its per-vCPU TLS id and
        // restore the caller's afterward — a real OS-thread child gets its own thread-local in `run_child`.
        let prev_tls = crate::vcpu_tls::get();
        crate::vcpu_tls::seed(vcpu_id);
        // A child that uses `cont.*` gets its own fiber execution context over the domain-shared table
        // (D57 3b-ii), like `run_child`; publish it as the current runtime for the run.
        let mut frt = env.fiber_cfg.map(|(tid, mask)| {
            let table = self
                .fiber_table()
                .expect("fiber_cfg set ⇒ the domain fiber table is set");
            let mut rt = FiberRuntime::new(table, tid, mask);
            rt.set_call_tramp(env.call_tramp);
            // Arm the durable fiber-switch swap for this child (slice 3.4): a child that owns fibers
            // needs `mem_base` + the `durable` flag on its own runtime so its `cont.resume` swap points
            // the active shadow-SP at the fiber's region, and so the freeze driver's `Complete` arm
            // records the flattened fiber as residue (it gates on the runtime's `durable`). The root's
            // runtime is armed in `run_inner`; a child's is built here, so arm it here.
            rt.set_durable_env(env.mem_base, env.durable);
            Box::new(rt)
        });
        let prev = frt
            .as_mut()
            .map(|b| fiber_rt::set_current(&mut **b as *mut FiberRuntime));

        let mut call = ChildCall {
            env,
            code,
            sp,
            arg,
            ret: 0,
        };
        // SAFETY: `child_entry` honours the `Entry` ABI; `call` outlives the call; the guarded call
        // nests cleanly. A guest fault unwinds back here and is reported as `MemoryFault` below.
        let faulted = mem::run_guarded_range(
            child_entry as *const () as *const u8,
            &mut call as *mut ChildCall as *const i64,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            env.fault_lo,
            env.fault_hi,
        );
        // Slice 3.4: a child that **owns fibers** must flatten the ones it parked into their shadow
        // regions — the JIT mirror of the interp's per-vCPU `freeze_drive`. The root's drive (in
        // `run_inner`) ran before this child existed, so it never saw these. Run it while the child's
        // runtime is still `CURRENT_RT` and the window is `UNWINDING`; `freeze_drive` resumes each
        // parked fiber to zero-progress completion, leaving its flattened extent in its slot and the
        // active shadow-SP restored to the child's region. Drain the child's residue into the domain
        // accumulator (`run_inner` merges it into the run's `frozen_out`; canonical sort-by-slot at
        // serialize keeps the artifact order-independent).
        if !faulted && fiber_rt::window_is_unwinding(env.mem_base) {
            if let Some(b) = frt.as_mut() {
                let rt = &mut **b as *mut FiberRuntime;
                fiber_rt::freeze_drive(rt, env.trap_out as u64);
                let child_frozen = fiber_rt::take_frozen(rt);
                if !child_frozen.is_empty() {
                    lock(&self.frozen_fibers).extend(child_frozen);
                }
            }
        }
        if let Some(pr) = prev {
            fiber_rt::set_current(pr);
        }
        crate::vcpu_tls::seed(prev_tls); // restore the caller (root) vCPU's TLS id

        let (result, trap) = if faulted {
            // SAFETY: live trap cell.
            store_trap(env.trap_out, TrapKind::MemoryFault as i64);
            (0, TrapKind::MemoryFault as i64)
        } else {
            let t = load_trap(env.trap_out);
            (call.ret, t)
        };
        (result, trap, faulted)
    }

    /// Run every child *deferred* during a durable freeze (slice 3.3), inline and in spawn order, once
    /// the root has unwound — the JIT's single-worker equivalent of the interpreter dispatching the
    /// enqueued children after the root yields. Each child runs in its own top-down shadow context
    /// (the active shadow-SP word points there for the run); one that unwound under the freeze records a
    /// [`crate::FrozenVCpu`] residue and keeps its context for thaw, while a genuine finish frees the
    /// context for reuse. The last child leaves the active shadow-SP at its own extent, matching the
    /// interp's dispatch-last convention so the window is byte-identical.
    ///
    /// # Safety
    /// Called from `run_inner` after the root's guarded call returned (no guest code is on this thread)
    /// on a durable freeze run; the env's window / fault range / call-trampoline are the live run's.
    pub(crate) unsafe fn drive_frozen_spawns(&self) {
        let env = self.env();
        // Arm this thread's detect-and-kill recovery for the inline runs (idempotent; the root's run
        // already installed the process-wide handler, but a child's `run_guarded_range` needs the
        // thread-local guard live — `install_guard` is a no-op if already armed).
        mem::install_guard();
        // **Loop-drain** the pending set (slice 3.4: nested spawns). A child running inline may itself
        // `defer_spawn` a grandchild, which lands in `pending_spawns` *after* this batch's snapshot — so
        // re-drain until empty. Each batch is a BFS level (children, then grandchildren, …), matching
        // the interpreter's runnable-queue order, so the freeze-time side-effect interleaving — and the
        // frozen window — is byte-identical.
        loop {
            let pending: Vec<PendingSpawn> = std::mem::take(&mut lock(&self.pending_spawns));
            if pending.is_empty() {
                break;
            }
            for p in pending {
                // §12.8 4A.5: this child's shadow-SP word lives in its **own** region (no shared word);
                // initialise it empty (frame base) and re-point `durable.shadow_base` so the child's
                // instrumented code addresses its own region during the unwind.
                let child_region = fiber_rt::shadow_region_base(p.ctx);
                fiber_rt::write_shadow_sp(
                    env.mem_base,
                    child_region,
                    fiber_rt::shadow_frame_base(p.ctx),
                );
                crate::durable_shadow::seed(child_region);
                // Attribute any grandchild this child spawns to it (its `parent_task` + per-vCPU table).
                *lock(&self.cur_task) = p.task;
                // §12: seed the child's per-vCPU TLS register to its (global) task id, matching the interp.
                let (result, trap, faulted) =
                    self.run_child_inline(env, p.code, p.sp, p.arg, p.task as i64);
                *lock(&self.cur_task) = 0; // back to the root between children
                crate::durable_shadow::seed(fiber_rt::shadow_region_base(0)); // back to the root's region

                // The child's flattened extent and whether it unwound under the freeze.
                let child_sp = fiber_rt::read_shadow_sp(env.mem_base, child_region);
                let froze = !faulted && fiber_rt::window_is_unwinding(env.mem_base);

                // A child that unwound under the freeze records *itself* as residue (its continuation now
                // lives in its own region; extent = the live shadow-SP) and keeps its context (re-spawned
                // there on thaw); its `parent_task` lets thaw rebuild the per-parent join topology. A
                // genuine finish frees the context for reuse (recycling).
                if froze {
                    lock(&self.frozen_vcpus).push(crate::FrozenVCpu {
                        task: p.task as usize,
                        parent_task: p.parent as usize,
                        func: p.func_idx as i32,
                        args: vec![p.sp as i64, p.arg as i64],
                        shadow_sp: child_sp,
                        completed_result: None, // a spilled (frozen) child re-runs on thaw
                    });
                } else if let Some(table) = self.fiber_table() {
                    table.free_vcpu_context(p.ctx);
                }

                // The child's computation has ended: free its concurrent-live slot, then publish the
                // result so a `thread.join` resolves it.
                {
                    let mut t = lock(&self.threads);
                    t.live -= 1;
                }
                let mut st = lock(&p.done.state);
                *st = Some((result, trap));
                p.done.cv.notify_all();
            }
        }
    }

    /// Re-attach and run the spawned children a multi-vCPU freeze flattened (slice 3.3, thaw side),
    /// **before** the root re-enters — the JIT's single-worker thaw. Mirrors the interp `drive` thaw
    /// re-spawn: the root's `REWINDING` rewind *skips* its prologue `thread.spawn` (it reloads the
    /// recorded handle), so a child that existed before the freeze point is reconstructed here, not by
    /// the root. Each child (in ascending task = spawn order) is registered into the join table at its
    /// own handle slot (`task − 1`, padding finished/joined slots so the root's reloaded handle still
    /// resolves) and run inline under `REWINDING` from its restored shadow extent: it rewinds, flips to
    /// `NORMAL`, runs forward to completion, and publishes its result — so the root's re-executed
    /// `thread.join` (after its checkpoint) resolves immediately. Finally the active shadow-SP + state
    /// word are set to the root's extent + `REWINDING` so the root rewinds from the right point.
    ///
    /// The children run *before* the root (rather than the interp's root-parks-on-join dispatch): a
    /// thaw runs to completion with no re-snapshot, and the §12.6 equivalence holds because a
    /// `REWINDING` vCPU **reloads** its recorded side effects (it never re-issues them), so the
    /// serialization order doesn't change the result. Scope mirrors the interp's: flat root-spawned
    /// children, no nested spawns, no child-owned fibers.
    ///
    /// # Safety
    /// Called from `run_inner` after `set_env`, before the root's guarded call, on a durable thaw run;
    /// the env's window / fault range / call-trampoline are the live run's, the reserve committed RW.
    pub(crate) unsafe fn thaw_reattach_and_run(&self, seed: &[crate::FrozenVCpu], root_sp: u64) {
        if seed.is_empty() {
            return;
        }
        let env = self.env();
        mem::install_guard();

        // (1) Rebuild the **per-parent** join tables in ascending task (= spawn) order — a parent's
        // table exists before a (grand)child attaches, and per-parent append reproduces the freeze-time
        // handles (slice 3.4: a grandchild's handle is its index in its *parent's* table, not a global
        // one). Each child is counted toward §15 live and keeps its Done cell + run params.
        struct Run {
            task: u64,
            parent: u64,
            func_idx: u32,
            ctx: usize,
            code: u64,
            sp: u64,
            arg: u64,
            shadow_sp: u64,
            done: std::sync::Arc<Done>,
        }
        let mut ordered: Vec<&crate::FrozenVCpu> = seed.iter().collect();
        ordered.sort_by_key(|v| v.task);
        let mut runs: Vec<Run> = Vec::with_capacity(ordered.len());
        for v in &ordered {
            // §12.8 4A.5 follow-up A: a child that **completed** before the freeze point (no frozen
            // continuation) gets its `thread.join` result delivered into the spawner's table directly —
            // its Done cell is pre-filled and it is **not** re-run (its side effects are already in the
            // snapshot). Pushed in task order alongside frozen children so handles stay dense.
            let done = std::sync::Arc::new(Done {
                state: Mutex::new(v.completed_result.map(|r| (r, 0))),
                cv: Condvar::new(),
            });
            {
                let mut dc = lock(&self.dchildren);
                let tbl = dc.entry(v.parent_task as u64).or_default();
                tbl.cells.push(std::sync::Arc::clone(&done));
                tbl.joined.push(false);
            }
            if v.completed_result.is_some() {
                continue; // already-done: no re-run, no §15 live count, no context
            }
            lock(&self.threads).live += 1;
            let entry = (env.fn_table_base as *const FnEntry).add(v.func as u32 as usize);
            runs.push(Run {
                task: v.task as u64,
                parent: v.parent_task as u64,
                func_idx: v.func as u32,
                ctx: fiber_rt::shadow_context_of_sp(v.shadow_sp),
                code: (*entry).code(),
                sp: v.args.first().copied().unwrap_or(0) as u64,
                arg: v.args.get(1).copied().unwrap_or(0) as u64,
                shadow_sp: v.shadow_sp,
                done,
            });
        }

        // (2) §12.8 concurrent-thaw stage 2: **re-spawn each frozen vCPU on its own OS thread**, rewinding
        // from its restored extent (`thaw_extent`) against its *own* per-context thaw word — concurrent
        // with its siblings and the root, mirroring a fresh concurrent durable run (vs. the former inline
        // serial "children before parents" loop). A re-issued `thread.join` blocks on the child's real
        // `Done` cell (filled when its thread publishes its result); a re-issued `atomic.wait`/`notify`
        // re-synchronises across the live threads (so a producer↔consumer pair frozen mid-rendezvous can
        // thaw). The root re-enters and rewinds after this returns; `run_inner`'s `join_all` joins the
        // children at run end. `run_child` does the per-context REWINDING setup, frees the context on a
        // genuine finish, drops the §15 live count, and publishes the result + notifies.
        for r in runs {
            let args = SpawnArgs {
                env,
                code: r.code,
                sp: r.sp,
                arg: r.arg,
                vcpu_id: r.task as i64,
                durable_child: Some(DurableChild {
                    task: r.task,
                    parent: r.parent,
                    ctx: r.ctx,
                    func_idx: r.func_idx,
                    thaw_extent: Some(r.shadow_sp), // rewind from the restored extent
                }),
                done: r.done,
                dom: self as *const Domain,
            };
            match std::thread::Builder::new()
                .name(format!("svm-thaw-{}", args.vcpu_id))
                .spawn(move || run_child(args))
            {
                Ok(jh) => lock(&self.threads).joins.push(jh),
                Err(_) => {
                    // Thread creation failed: undo the §15 live count (taken above) and free the context.
                    // A thaw that can't re-spawn its children is already fatal (the root's join will hang),
                    // but keep the accounting consistent.
                    lock(&self.threads).live -= 1;
                    if let Some(table) = self.fiber_table() {
                        table.free_vcpu_context(r.ctx);
                    }
                }
            }
        }
        // The root rewinds first on its re-entry: point the active shadow-SP at its restored extent and
        // re-arm REWINDING (the last child flipped the word to NORMAL when its rewind completed).
        // §12.8 4A.5: the root's extent goes into context 0's own region word.
        fiber_rt::write_shadow_sp(env.mem_base, fiber_rt::shadow_region_base(0), root_sp);
        crate::durable_shadow::seed(fiber_rt::shadow_region_base(0));
        fiber_rt::window_set_rewinding(env.mem_base, 0); // the root's own (ctx 0) thaw word

        *lock(&self.cur_task) = 0; // the root runs next (its joins resolve in its table)
    }
}

/// `thread.join` thunk: block this OS thread until vCPU `handle` finishes; return its `i64` result. A
/// forged / out-of-range handle is inert (**traps** `ThreadFault`); a child that itself trapped
/// propagates that trap here (the joiner's `emit_trap_propagate` then unwinds).
///
/// # Safety
/// `sched` is the run's live `Domain`; `trap_out` is the live trap cell.
pub(crate) unsafe extern "C" fn thread_join(
    sched: *const Domain,
    handle: i32,
    trap_out: u64,
) -> i64 {
    let dom = &*sched;
    // Durable single-worker (slice 3.4): resolve the handle in the **current** vCPU's per-vCPU table —
    // nested spawns give each spawning vCPU its own handle namespace. `dchildren` is populated only on
    // that path (freeze defer / thaw re-attach); when it's empty this is the global OS-thread table.
    // §12.8 concurrent-thaw stage 2: on a concurrent durable child's OS thread, use *this* thread's task
    // (set by `run_child`) — concurrent children would race the shared `cur_task`. The root's own thread
    // has no `CONCURRENT_SPAWN_TASK`, so it falls back to `cur_task` (set to the root by the thaw driver).
    let cur = CONCURRENT_SPAWN_TASK
        .with(|c| c.get())
        .unwrap_or_else(|| *lock(&dom.cur_task));
    let done = {
        let mut dc = lock(&dom.dchildren);
        if !dc.is_empty() {
            // Per-vCPU: the current vCPU's children. A vCPU with no table (spawned nothing) ⇒ a forged
            // handle ⇒ inert trap.
            match dc.get_mut(&cur) {
                Some(tbl) => {
                    let n = tbl.cells.len();
                    let mask = if n == 0 { 0 } else { n.next_power_of_two() - 1 };
                    let slot = (handle as u32 as usize) & mask;
                    if slot >= n || tbl.joined[slot] {
                        store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                        return 0;
                    }
                    tbl.joined[slot] = true;
                    std::sync::Arc::clone(&tbl.cells[slot])
                }
                None => {
                    store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                    return 0;
                }
            }
        } else {
            drop(dc);
            let mut t = lock(&dom.threads);
            let n = t.cells.len();
            let mask = if n == 0 { 0 } else { n.next_power_of_two() - 1 };
            let slot = (handle as u32 as usize) & mask;
            // Out-of-range or already-joined handles are inert (trap), like a forged capability index.
            if slot >= n || t.joined[slot] {
                store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
                return 0;
            }
            t.joined[slot] = true;
            std::sync::Arc::clone(&t.cells[slot])
        }
    };
    // §5 kill-path: a joiner blocked on a sibling must also unwind when the host kills the domain
    // (else `join_all` hangs on a vCPU that will never finish). When armed, re-check the interrupt
    // cell periodically; on a kill, return so the caller's next epoch poll traps `OutOfFuel`.
    let epoch_addr = dom.env().epoch_addr;
    // §12.8 4A.5 (blocked-in-join freeze): a joiner parked here when a freeze begins must also return
    // so its instrumented caller can unwind at the re-issue safepoint the `svm-durable` transform now
    // emits after `thread.join`. On thaw the join is re-issued and re-parks on the (re-spawned) child.
    // Gate on `durable` — only then is window offset 0 the reserved state word; a non-durable run's
    // offset 0 is ordinary guest memory and could spuriously read `UNWINDING`.
    let unwind_base = if dom.env().durable {
        dom.env().mem_base
    } else {
        0
    };
    let mut st = lock(&done.state);
    // §12.8 concurrent-thaw stage 3: count this joiner as blocked while it parks, so a sibling waiter's
    // `peers_live` (and the deadlock detector) see a join↔wait mutual block as full quiescence.
    let _pg = ParkGuard::new(&dom.parked);
    loop {
        if let Some((result, trap)) = *st {
            if trap != 0 {
                store_trap(trap_out as *mut i64, trap);
            }
            return result;
        }
        if epoch_fired(epoch_addr) {
            return 0; // killed — unwind to guest code, which traps OutOfFuel at its next poll
        }
        // SAFETY: on a durable run `mem_base` is the committed window base, offset 0 RW for the run.
        if unwind_base != 0 && unsafe { fiber_rt::window_is_unwinding(unwind_base) } {
            return 0; // freeze in progress — return so the join's trailing safepoint unwinds
        }
        // A timeout wait so the kill (`epoch_addr`) and freeze (`unwind_base`) re-checks above run
        // periodically even when no `notify` arrives; a plain `wait` only when neither is armed.
        #[cfg(not(loom))]
        if epoch_addr != 0 || unwind_base != 0 {
            st = done
                .cv
                .wait_timeout(st, KILL_RECHECK)
                .unwrap_or_else(|e| e.into_inner())
                .0;
            continue;
        }
        st = done.cv.wait(st).unwrap_or_else(|e| e.into_inner());
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

/// Read the `width`-byte value at confined physical address `phys` (the lowering's alignment guard
/// ensures it is aligned).
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

/// `<ty>.atomic.wait` thunk: if the `width`-byte value at confined `phys` still equals `expected`, park
/// this OS thread on `phys` until a `notify` or `timeout` ns elapse (`< 0` = forever). Returns the
/// `i32` status (woken / not-equal / timed-out). The value is re-read **under the futex lock**, so a
/// concurrent store-then-`notify` cannot be lost.
///
/// §12.8 parked-vCPU slice: like `thread_join`, a waiter parked here when a freeze begins returns on
/// observing `UNWINDING` (its instrumented caller then unwinds at the trailing re-issue safepoint). On a
/// **thaw** (stage 2) the wait is re-issued and runs concurrently with its re-spawned siblings: a wake
/// that landed as a value change resolves immediately (`WAIT_NOT_EQUAL`, no park); a re-issue that parks
/// is woken by a sibling's re-issued `notify` (the producer↔consumer rendezvous). A wait whose every
/// possible notifier has exited fails closed via `trap_out` (`ThreadFault`) by the shared deadlock
/// detection in [`futex_wait`] (no thaw-specific path; `WAIT_DEADLOCK` is intercepted below).
///
/// # Safety
/// `sched` is the run's live `Domain`; `phys` points at `width` readable guest bytes; `trap_out` is the
/// live trap cell.
pub(crate) unsafe extern "C" fn thread_wait(
    sched: *const Domain,
    phys: u64,
    expected: u64,
    width: u32,
    timeout: i64,
    trap_out: u64,
) -> i32 {
    let dom = &*sched;
    let mask = width_mask(width);
    let deadline = if timeout < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_nanos(timeout as u64))
    };
    // §12.8 parked-vCPU freeze: gate on `durable` so window offset 0 is the reserved state word (a
    // non-durable run's offset 0 is ordinary guest memory that could spuriously read `UNWINDING`).
    let unwind_base = if dom.env().durable {
        dom.env().mem_base
    } else {
        0
    };
    let status = futex_wait(
        &dom.futex,
        &dom.futex_cv,
        phys,
        || read_phys(phys, width) & mask == expected & mask,
        deadline,
        dom.env().epoch_addr,
        unwind_base,
        &dom.parked,
        // §12.8 concurrent-thaw stage 3: a notifier must be a live vCPU that is **not** itself parked.
        // `live` counts the root + unfinished spawns (incl. this waiter); `parked` counts those blocked in
        // wait/join (incl. this waiter). `live > parked` ⇒ some live vCPU is still runnable and could
        // notify; `live == parked` ⇒ every live vCPU is blocked (a lone or **mutual** deadlock).
        || lock(&dom.threads).live > dom.parked.load(Ordering::Acquire),
    );
    // §12.8 concurrent-thaw stage 2: an infinite wait with no possible notifier left is a guest deadlock —
    // surface it as `ThreadFault` (matching the interpreter), never as a guest-visible wait status.
    #[cfg(not(loom))]
    if status == WAIT_DEADLOCK {
        store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
        return 0;
    }
    status
}

/// `atomic.notify` thunk: wake up to `count` vCPUs parked on confined `phys`; return the `i32` count
/// woken.
///
/// # Safety
/// `sched` is the run's live `Domain`.
pub(crate) unsafe extern "C" fn thread_notify(sched: *const Domain, phys: u64, count: i32) -> i32 {
    let dom = &*sched;
    // The count is **unsigned** "wake up to N" (wasm's notify count is u32; `-1` = wake all);
    // `futex_notify` caps at the real waiter count, so reinterpret the i32 bits as u32.
    futex_notify(&dom.futex, &dom.futex_cv, phys, count as u32) as i32
}

/// §12.8 concurrent-thaw stage 3: RAII counter for [`Domain::parked`]. Increments while a vCPU is blocked
/// (a `atomic.wait` park or a `thread.join` park) and decrements on **every** exit path (woken, freeze,
/// kill, deadlock, child-done). The deadlock detector reads the count to distinguish "all live vCPUs are
/// blocked" (quiescence ⇒ no notifier can run ⇒ deadlock) from "a notifier is still runnable".
struct ParkGuard<'a>(&'a AtomicUsize);
impl<'a> ParkGuard<'a> {
    fn new(parked: &'a AtomicUsize) -> Self {
        parked.fetch_add(1, Ordering::AcqRel);
        ParkGuard(parked)
    }
}
impl Drop for ParkGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Futex park core (shared by the thunk and the loom test). `still_eq` re-checks the guest value under
/// the lock; returns a `WAIT_*` status. Spurious wakeups are spec-allowed, but the per-`key` generation
/// makes a real `notify` distinguishable so the returned status is accurate.
#[allow(clippy::too_many_arguments)] // a futex park threads its full guest/kill/freeze/deadlock context
fn futex_wait(
    futex: &Mutex<HashMap<u64, FutexEntry>>,
    cv: &Condvar,
    key: u64,
    still_eq: impl Fn() -> bool,
    deadline: Option<Instant>,
    epoch_addr: usize,
    unwind_base: u64,
    // §12.8 concurrent-thaw stage 3: count of vCPUs currently blocked (wait/join). This waiter joins it
    // for its park (so `peers_live` sees it), via the `ParkGuard` below.
    parked: &AtomicUsize,
    // §12.8 concurrent-thaw stage 2/3: `true` while some live vCPU is **not** parked (could still run to
    // a `notify`). An infinite wait whose `peers_live()` is false (every live vCPU is blocked) can never
    // be satisfied → [`WAIT_DEADLOCK`].
    peers_live: impl Fn() -> bool,
) -> i32 {
    let mut g = lock(futex);
    if !still_eq() {
        return WAIT_NOT_EQUAL;
    }
    let start_gen = {
        let e = g.entry(key).or_default();
        e.waiters += 1;
        e.generation
    };
    // Count this vCPU as blocked for the duration of the park (dropped on every loop exit below), so a
    // peer's `peers_live` — and the deadlock check below — see it.
    let _pg = ParkGuard::new(parked);
    let status = loop {
        let cur = g.get(&key).map(|e| e.generation).unwrap_or(start_gen);
        if cur != start_gen {
            break WAIT_WOKEN;
        }
        // §5 kill-path: a parked waiter unwinds when the host kills the domain (it returns as if
        // woken; the guest code after the wait traps `OutOfFuel` at its next epoch poll).
        if epoch_fired(epoch_addr) {
            break WAIT_WOKEN;
        }
        // §12.8 parked-vCPU freeze: a waiter parked here when a freeze begins must also return so its
        // instrumented caller can unwind at the re-issue safepoint the `svm-durable` transform now emits
        // after `atomic.wait`. The returned status is discarded (the safepoint unwinds before the guest
        // observes it); on thaw the wait is re-issued. SAFETY: on a durable run `unwind_base` is the
        // committed window base, offset 0 RW for the run.
        #[cfg(not(loom))]
        if unwind_base != 0 && unsafe { fiber_rt::window_is_unwinding(unwind_base) } {
            break WAIT_WOKEN;
        }
        match deadline {
            None => {
                // Armed (real build): bounded re-check so an *infinite* wait still observes a kill or a
                // freeze even when no `notify` arrives.
                #[cfg(not(loom))]
                if epoch_addr != 0 || unwind_base != 0 {
                    // §12.8 concurrent-thaw stage 2: deadlock detection. If no other vCPU is live, no
                    // `notify` can ever arrive (a parked waiter can't notify itself, and a wasm wait
                    // returns only on notify/timeout — not on a plain value change), so this infinite wait
                    // can never be satisfied. Fail closed rather than re-check forever. Detected within
                    // `KILL_RECHECK` of the last peer exiting (run_child drops `live` as each finishes).
                    if !peers_live() {
                        break WAIT_DEADLOCK;
                    }
                    g = cv
                        .wait_timeout(g, KILL_RECHECK)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                    continue;
                }
                // §12.8 concurrent-thaw stage 3: the same deadlock check, modeled under loom. loom has no
                // timeouts (so the real-build `wait_timeout` re-check above is compiled out), but it *does*
                // explore the peer-exit↔consumer-wait interleavings: a peer that goes non-live notifies
                // this `cv`, and we re-evaluate `peers_live` on each wakeup, so an infinite wait with no
                // possible notifier resolves to `WAIT_DEADLOCK` instead of blocking the model forever.
                #[cfg(loom)]
                if !peers_live() {
                    break WAIT_DEADLOCK;
                }
                g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            // loom's `Condvar` models no timeouts; the loom test only exercises the infinite-wait path
            // (`deadline = None`), so the timed branch is real-build-only.
            #[cfg(not(loom))]
            Some(dl) => {
                let now = Instant::now();
                if now >= dl {
                    break WAIT_TIMED_OUT;
                }
                let (ng, to) = cv
                    .wait_timeout(g, dl - now)
                    .unwrap_or_else(|e| e.into_inner());
                g = ng;
                if to.timed_out() {
                    let cur = g.get(&key).map(|e| e.generation).unwrap_or(start_gen);
                    break if cur != start_gen {
                        WAIT_WOKEN
                    } else {
                        WAIT_TIMED_OUT
                    };
                }
            }
            #[cfg(loom)]
            Some(_dl) => unreachable!("loom futex model uses no timeout"),
        }
    };
    if let Some(e) = g.get_mut(&key) {
        e.waiters = e.waiters.saturating_sub(1);
        // Audit #8: drop a fully-drained entry so the futex map can't accumulate stale keys. Safe
        // under the held lock — `waiters == 0` means no one is parked, so the per-key generation
        // has no live observer to preserve; a later waiter on this key starts a fresh entry.
        if e.waiters == 0 {
            g.remove(&key);
        }
    }
    status
}

/// Futex wake core: bump `key`'s generation (so up to `count` parked waiters observe a real wake) and
/// `notify_all`; return how many waiters were parked (capped at `count`).
fn futex_notify(
    futex: &Mutex<HashMap<u64, FutexEntry>>,
    cv: &Condvar,
    key: u64,
    count: u32,
) -> u32 {
    let woken = {
        let mut g = lock(futex);
        match g.get_mut(&key) {
            Some(e) if e.waiters > 0 && count > 0 => {
                e.generation = e.generation.wrapping_add(1);
                e.waiters.min(count)
            }
            _ => 0,
        }
    };
    if woken > 0 {
        cv.notify_all();
    }
    woken
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::{AtomicU64, Ordering};
    use loom::sync::Arc;

    /// notify must never be lost: with one waiter and one notifier racing on a futex word, the waiter
    /// always ends up woken (never parked forever), whether the store+notify lands before or after the
    /// waiter parks. Mirrors the property the old `par` worker-pool loom test checked, now for the 1:1
    /// futex primitive.
    #[test]
    fn loom_wait_notify_never_hangs() {
        loom::model(|| {
            let futex = Arc::new(Mutex::new(HashMap::<u64, FutexEntry>::new()));
            let cv = Arc::new(Condvar::new());
            let word = Arc::new(AtomicU64::new(0)); // the guest futex word
            const KEY: u64 = 0x1000;

            let (f2, cv2, w2) = (Arc::clone(&futex), Arc::clone(&cv), Arc::clone(&word));
            let producer = loom::thread::spawn(move || {
                // store then notify (the release/wake pair)
                w2.store(1, Ordering::SeqCst);
                futex_notify(&f2, &cv2, KEY, 1);
            });

            // consumer: wait while the word is still 0. Must not hang: either it sees 1 (NOT_EQUAL) or
            // it parks and the notify wakes it (WOKEN). A finite deadline keeps the model bounded, but
            // the invariant is "doesn't time out".
            let parked = AtomicUsize::new(0); // unread here: the model's `peers_live` is always true
            let status = futex_wait(
                &futex,
                &cv,
                KEY,
                || word.load(Ordering::SeqCst) == 0,
                None,
                0,
                0,
                &parked,
                || true, // loom models the producer↔consumer rendezvous; a peer is always live
            );
            producer.join().unwrap();
            assert!(status == WAIT_WOKEN || status == WAIT_NOT_EQUAL);
        });
    }

    /// §12.8 concurrent-thaw stage 3: the mutual-deadlock detection must **resolve, never hang the
    /// model**, when the last possible notifier exits. A consumer waits on a word that never changes; a
    /// "peer" (its only possible notifier) goes non-live and signals the cv. Under every interleaving
    /// (peer exits before *or* after the consumer parks) the consumer returns `WAIT_DEADLOCK`. This is
    /// the loom analogue of the real-build `live > parked` quiescence check — here `peers_live` reads a
    /// modeled live-peer flag. The peer flips the flag + wakes **under the futex lock**, exactly as
    /// `futex_notify` serializes a wake, so the consumer's check-then-park can never miss the transition.
    #[test]
    fn loom_deadlock_detection_resolves_when_last_peer_exits() {
        loom::model(|| {
            let futex = Arc::new(Mutex::new(HashMap::<u64, FutexEntry>::new()));
            let cv = Arc::new(Condvar::new());
            // A loom atomic (explored by the model); `parked` below stays the std type `futex_wait` takes.
            let peer_live = Arc::new(loom::sync::atomic::AtomicUsize::new(1)); // 1 ⇒ a peer could notify
            const KEY: u64 = 0x2000;

            let (f2, cv2, pl2) = (Arc::clone(&futex), Arc::clone(&cv), Arc::clone(&peer_live));
            let peer = loom::thread::spawn(move || {
                // The peer finishes its computation (can no longer notify). Serialize the state change +
                // wake under the futex lock, like `futex_notify`, so the consumer can't lose it.
                let _g = f2.lock().unwrap_or_else(|e| e.into_inner());
                pl2.store(0, Ordering::SeqCst);
                cv2.notify_all();
            });

            let parked = AtomicUsize::new(0);
            let status = futex_wait(
                &futex,
                &cv,
                KEY,
                || true, // the guest word never changes (always still equals `expected`)
                None,
                0,
                0,
                &parked,
                || peer_live.load(Ordering::SeqCst) > 0, // a live peer could still notify
            );
            peer.join().unwrap();
            assert_eq!(
                status, WAIT_DEADLOCK,
                "no possible notifier left ⇒ deadlock detected, not an infinite block",
            );
        });
    }

    /// O-A4 (Phase-4 Slice A): the multi-worker quiesce barrier — the R10 seam. §12.8 4A.5: each worker
    /// unwinds into its **own** per-context region's shadow-SP word (no shared scratch since the
    /// relocation), so the unwind runs concurrently *outside* the lock; the lock guards only the join.
    /// `N` workers each unwind + arrive; the coordinator waits for all. Under every interleaving:
    ///   * O-A4.2/.3 — each worker's own SP slot ends at exactly the extent it wrote (concurrent
    ///     unwinds into disjoint per-context words never clobber a sibling);
    ///   * O-A4.4 — the coordinator never hangs (the last-arriver notify can't be lost — same property
    ///     as `loom_wait_notify_never_hangs`, here for the barrier).
    /// The single-vCPU async-request ordering (O-A3) is the degenerate `N = 1` case.
    #[test]
    fn loom_quiesce_barrier_never_hangs_with_per_context_sp() {
        loom::model(|| {
            const N: usize = 2;
            let dom = Arc::new(Domain::new(MAX_VCPUS));
            dom.arm_quiesce(N);
            // §12.8 4A.5: per-context SP words — each worker owns its own slot (no shared scratch).
            let sps: Vec<Arc<AtomicU64>> = (0..N).map(|_| Arc::new(AtomicU64::new(0))).collect();

            let workers: Vec<_> = (0..N)
                .map(|i| {
                    let (d, sp) = (Arc::clone(&dom), Arc::clone(&sps[i]));
                    loom::thread::spawn(move || {
                        let my_extent = (i as u64) + 1; // a distinct per-context extent
                        d.quiesce_arrive(|| {
                            // The unwind runs concurrently with siblings: it touches only this
                            // context's own region word, so there is no critical section to serialize.
                            sp.store(my_extent, Ordering::Release);
                        });
                    })
                })
                .collect();

            // The coordinator waits for every worker to quiesce — must not hang under any interleaving.
            dom.quiesce_wait_all();
            for w in workers {
                w.join().unwrap();
            }
            // O-A4.2/.3: each context's own word holds exactly its extent — concurrent unwinds into
            // disjoint per-context words never clobbered a sibling.
            for (i, sp) in sps.iter().enumerate() {
                assert_eq!(
                    sp.load(Ordering::Acquire),
                    (i as u64) + 1,
                    "a context's own per-context SP word was clobbered"
                );
            }
        });
    }
}
