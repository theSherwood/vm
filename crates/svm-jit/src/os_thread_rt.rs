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

use crate::fiber_rt::{self, FiberCallTramp, FiberRuntime};
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
    /// its own `FiberRuntime` for `cont.*`. `None` for pure-thread modules.
    fiber_cfg: Option<(u32, u64, usize)>,
    /// Address of the §5 kill-path interrupt cell (an `AtomicU64`), or `0` when no kill-path is
    /// armed. *Spinning* vCPUs already poll it (it is baked into the same compiled code every vCPU
    /// runs); this lets a **parked** vCPU — blocked in a futex `wait` or `thread.join` — re-check it
    /// and unwind too, so one host interrupt kills the whole domain rather than only its busy threads.
    epoch_addr: usize,
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
    threads: Mutex<Threads>,
    futex: Mutex<HashMap<u64, FutexEntry>>,
    futex_cv: Condvar,
    /// §15 spawn quota: max **concurrently-live** vCPUs (incl. the root) this domain may have, clamped
    /// to [`MAX_VCPUS`]. Exceeding it is a clean `ThreadFault`. Bounds `Threads::live` (concurrent),
    /// matching the interpreter's `s.live` — a spawn-join loop is fine (a finished vCPU frees its slot).
    max_vcpus: usize,
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
            // `live` starts at 1: the root vCPU (the main thread running the entry) counts toward the
            // §15 quota, like the interpreter's `s.live`.
            threads: Mutex::new(Threads {
                live: 1,
                ..Threads::default()
            }),
            futex: Mutex::new(HashMap::new()),
            futex_cv: Condvar::new(),
            max_vcpus: max_vcpus.clamp(1, MAX_VCPUS),
        }
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
        fiber_cfg: Option<(u32, u64, usize)>,
        epoch_addr: usize,
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
        });
    }

    fn env(&self) -> Env {
        lock(&self.env).expect("Domain::set_env before any thread op")
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
    done: std::sync::Arc<Done>,
    /// The owning [`Domain`] — so this vCPU can drop its §15 concurrent-live count when it finishes.
    /// The domain outlives every spawned thread (`run_inner` joins them at run end), so the pointer
    /// stays valid for the thread's lifetime.
    dom: *const Domain,
}
// SAFETY: same contract as `Env` — the raw pointers are the run's shared window/trap cell, and a fresh
// OS thread is the sole user of its `SpawnArgs` until it stores into the (synchronized) `Done` cell.
unsafe impl Send for SpawnArgs {}

fn run_child(a: SpawnArgs) {
    let env = a.env;
    // Arm this OS thread's detect-and-kill recovery (idempotent; handler is process-wide, recovery is
    // thread-local — §5 / `mem::install_guard`).
    mem::install_guard();
    // A vCPU that uses `cont.*` gets its own fiber runtime, published for the duration of its run.
    let mut frt = env.fiber_cfg.map(|(tid, mask, max_fibers)| {
        let mut rt = FiberRuntime::new(tid, mask, max_fibers);
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
    let done = {
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
    futex_notify(&dom.futex, &dom.futex_cv, phys, count.max(0) as u32) as i32
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
