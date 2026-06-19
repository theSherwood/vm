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
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
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
    trap_capture: Mutex<Option<(usize, Vec<usize>)>>,
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
            frozen_fibers: Mutex::new(Vec::new()),
            dchildren: Mutex::new(HashMap::new()),
            cur_task: Mutex::new(0),
            next_task: Mutex::new(1), // root is task 0; first spawn is 1
            futex: Mutex::new(HashMap::new()),
            futex_cv: Condvar::new(),
            max_vcpus: max_vcpus.clamp(1, MAX_VCPUS),
            trap_capture: Mutex::new(None),
        }
    }

    /// Publish a **spawned** vCPU's raw trap-time backtrace capture (§5 W3 Stage 3), last-wins. Called
    /// from a dying worker ([`run_child`]) when it trapped, so the run thread can symbolize a trap that
    /// originated off the run thread (whose own thread-local capture it can't see).
    pub(crate) fn publish_trap_capture(&self, cap: (usize, Vec<usize>)) {
        *lock(&self.trap_capture) = Some(cap);
    }

    /// Take the trap-time backtrace capture a spawned vCPU published (§5 W3 Stage 3), if any. Called
    /// on the run thread after [`Self::join_all`], so the publishing worker has finished.
    pub(crate) fn take_trap_capture(&self) -> Option<(usize, Vec<usize>)> {
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

    /// Drain the durable freeze residue collected by inline children (slice 3.3) — called by
    /// `run_inner` after the root unwinds, so the durable entry can hand it to the embedder for the
    /// snapshot. Empty on a non-durable / non-freeze run.
    pub(crate) fn take_frozen_vcpus(&self) -> Vec<crate::FrozenVCpu> {
        std::mem::take(&mut lock(&self.frozen_vcpus))
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
struct SpawnArgs {
    env: Env,
    code: u64,
    sp: u64,
    arg: u64,
    /// §12 dense vCPU id seeded into this child's per-vCPU TLS register (`vcpu.tls`). Root is 0, so a
    /// spawned vCPU takes `handle + 1` (handles are 0-based, cumulative spawn order).
    vcpu_id: i64,
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

fn run_child(a: SpawnArgs) {
    let env = a.env;
    // §12 seed this vCPU's per-vCPU TLS register to its dense id before any guest code runs.
    crate::vcpu_tls::seed(a.vcpu_id);
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
    let done = std::sync::Arc::new(Done {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    let handle = {
        let mut t = lock(&dom.threads);
        // §15: bound *concurrent* live vCPUs (root + unfinished spawns), not the cumulative handle
        // table — so a spawn-join loop never trips, matching the interpreter.
        if t.live >= dom.max_vcpus {
            store_trap(trap_out as *mut i64, TrapKind::ThreadFault as i64);
            return -1;
        }
        let idx = t.cells.len();
        t.cells.push(std::sync::Arc::clone(&done));
        t.joined.push(false);
        let args = SpawnArgs {
            env,
            code,
            sp,
            arg,
            vcpu_id: idx as i64 + 1, // dense id seed (root 0; children 1, 2, …)
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
                // Out of OS threads: pop the cell we reserved and trap (no `live` change).
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
                // Point the active shadow-SP at this child's region so its unwind spills into its own
                // context; a later child overwrites the word, so the last leaves it at its extent.
                fiber_rt::write_shadow_sp(env.mem_base, fiber_rt::shadow_region_base(p.ctx));
                // Attribute any grandchild this child spawns to it (its `parent_task` + per-vCPU table).
                *lock(&self.cur_task) = p.task;
                // §12: seed the child's per-vCPU TLS register to its (global) task id, matching the interp.
                let (result, trap, faulted) =
                    self.run_child_inline(env, p.code, p.sp, p.arg, p.task as i64);
                *lock(&self.cur_task) = 0; // back to the root between children

                // The child's flattened extent and whether it unwound under the freeze.
                let child_sp = fiber_rt::read_shadow_sp(env.mem_base);
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
            let done = std::sync::Arc::new(Done {
                state: Mutex::new(None),
                cv: Condvar::new(),
            });
            {
                let mut dc = lock(&self.dchildren);
                let tbl = dc.entry(v.parent_task as u64).or_default();
                tbl.cells.push(std::sync::Arc::clone(&done));
                tbl.joined.push(false);
            }
            lock(&self.threads).live += 1;
            let entry = (env.fn_table_base as *const FnEntry).add(v.func as u32 as usize);
            runs.push(Run {
                task: v.task as u64,
                ctx: fiber_rt::shadow_context_of_sp(v.shadow_sp),
                code: (*entry).code(),
                sp: v.args.first().copied().unwrap_or(0) as u64,
                arg: v.args.get(1).copied().unwrap_or(0) as u64,
                shadow_sp: v.shadow_sp,
                done,
            });
        }

        // (2) Run children in **descending** task order — children *before* parents — so a parent's
        // re-executed `thread.join` finds an already-completed child (the JIT can't park-and-resume a
        // parent on the single worker; the interp parks it instead). A `REWINDING` vCPU **reloads** its
        // recorded side effects, so this order can't change the result (§12.6). Each runs under
        // `REWINDING` from its restored extent, completes (a basic thaw doesn't re-freeze), frees its
        // context, and publishes its result into its parent's table cell.
        runs.sort_by_key(|r| std::cmp::Reverse(r.task));
        for r in &runs {
            fiber_rt::window_set_rewinding(env.mem_base);
            fiber_rt::write_shadow_sp(env.mem_base, r.shadow_sp);
            *lock(&self.cur_task) = r.task; // route its own grandchild joins to its table
                                            // §12: seed the child's per-vCPU TLS register to its task id (matching the interp).
            let (result, trap, _faulted) =
                self.run_child_inline(env, r.code, r.sp, r.arg, r.task as i64);
            *lock(&self.cur_task) = 0;
            if let Some(table) = self.fiber_table() {
                table.free_vcpu_context(r.ctx);
            }
            lock(&self.threads).live -= 1;
            let mut st = lock(&r.done.state);
            *st = Some((result, trap));
            r.done.cv.notify_all();
        }
        // The root rewinds first on its re-entry: point the active shadow-SP at its restored extent and
        // re-arm REWINDING (the last child flipped the word to NORMAL when its rewind completed).
        fiber_rt::write_shadow_sp(env.mem_base, root_sp);
        fiber_rt::window_set_rewinding(env.mem_base);
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
    let cur = *lock(&dom.cur_task);
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
    let mut st = lock(&done.state);
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
        #[cfg(not(loom))]
        if epoch_addr != 0 {
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
/// # Safety
/// `sched` is the run's live `Domain`; `phys` points at `width` readable guest bytes.
pub(crate) unsafe extern "C" fn thread_wait(
    sched: *const Domain,
    phys: u64,
    expected: u64,
    width: u32,
    timeout: i64,
) -> i32 {
    let dom = &*sched;
    let mask = width_mask(width);
    let deadline = if timeout < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_nanos(timeout as u64))
    };
    futex_wait(
        &dom.futex,
        &dom.futex_cv,
        phys,
        || read_phys(phys, width) & mask == expected & mask,
        deadline,
        dom.env().epoch_addr,
    )
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

/// Futex park core (shared by the thunk and the loom test). `still_eq` re-checks the guest value under
/// the lock; returns a `WAIT_*` status. Spurious wakeups are spec-allowed, but the per-`key` generation
/// makes a real `notify` distinguishable so the returned status is accurate.
fn futex_wait(
    futex: &Mutex<HashMap<u64, FutexEntry>>,
    cv: &Condvar,
    key: u64,
    still_eq: impl Fn() -> bool,
    deadline: Option<Instant>,
    epoch_addr: usize,
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
        match deadline {
            None => {
                // Armed (real build): bounded re-check so an *infinite* wait still observes a kill.
                #[cfg(not(loom))]
                if epoch_addr != 0 {
                    g = cv
                        .wait_timeout(g, KILL_RECHECK)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                    continue;
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
            let status = futex_wait(
                &futex,
                &cv,
                KEY,
                || word.load(Ordering::SeqCst) == 0,
                None,
                0,
            );
            producer.join().unwrap();
            assert!(status == WAIT_WOKEN || status == WAIT_NOT_EQUAL);
        });
    }
}
