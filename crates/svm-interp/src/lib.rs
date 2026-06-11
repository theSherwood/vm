//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use svm_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, Data, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func,
    FuncIdx, FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Memory, Module, StoreOp, Terminator,
    ValIdx, ValType, DEFAULT_RESERVED_LOG2,
};
use svm_mask::Window;
use svm_mem::{Region, RmwOp};

/// A runtime value. Mirrors `ValType`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Reasons execution stopped without producing results.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Trap {
    /// Ran out of fuel (potential infinite loop) — see `run`.
    OutOfFuel,
    /// Integer division or remainder by zero (§3b).
    DivByZero,
    /// Signed `div_s` of `INT_MIN / -1`: the quotient `+2^31` is not representable, so
    /// it traps (§3b: trap only when there is no representable result). `rem_s` does
    /// **not** trap here — the remainder `0` *is* representable.
    IntOverflow,
    /// A memory access crossed the top of the window (guard-region fault, §4/§5).
    MemoryFault,
    /// Call recursion exceeded the interpreter's depth bound (host-stack guard).
    StackOverflow,
    /// `call_indirect` selected an empty table slot or a function whose signature
    /// did not match the call's type (the §3c table type-id check).
    IndirectCallType,
    /// Reached an `unreachable`/`trap` terminator (§3b).
    Unreachable,
    /// A trapping float→int conversion saw NaN or an out-of-range value (§3b).
    BadConversion,
    /// A `cap.call` named a handle that is forged, closed/revoked (dead generation),
    /// or the wrong interface type — the index was **inert** (§3c). Not an escape.
    CapFault,
    /// The guest invoked the `Exit` capability; carries the requested exit code. Not
    /// an error — the domain asked to terminate (§3e). Propagates like a trap.
    Exit(i32),
    /// A §12 fiber operation failed: `cont.resume` named a forged/dead/already-running
    /// fiber handle (inert, like [`Trap::CapFault`]), `suspend` ran at the root (no fiber
    /// to suspend to), or the fiber count exceeded the interpreter's bound. Not an escape.
    FiberFault,
    /// A §12 thread operation failed: `thread.join` named a forged / out-of-range / already-joined
    /// thread handle (inert, like [`Trap::CapFault`]), or `thread.spawn` exceeded the run's thread
    /// budget. Not an escape.
    ThreadFault,
    /// Structurally invalid in a way a verified module never is (defensive only).
    Malformed,
}

/// Maximum nested `call` depth before the interpreter traps, bounding the size of the
/// **explicit** guest call stack (a `Vec<Frame>`, §12) so adversarial (or merely deep)
/// guest recursion yields a clean `Trap::StackOverflow` rather than unbounded growth.
///
/// The interpreter no longer recurses on the host stack — the guest call stack is
/// reified (so a fiber's continuation is just its `Vec<Frame>`, suspendable; §12), and
/// the host stack stays O(1) regardless of guest depth. This is a reference-oracle limit,
/// not the production recursion ceiling (the JIT uses the guest's guard-paged data stack,
/// §5).
const MAX_CALL_DEPTH: u32 = 256;

/// Run `func` with `args`, consuming up to `*fuel` execution steps.
///
/// Returns the function's result values, or a `Trap`. Decrements `*fuel` per
/// instruction and per branch so that even an infinite loop terminates — important
/// for fuzzing and for never hanging a test.
pub fn run(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    // No capabilities granted: an empty powerbox (any `cap.call` is inert → `CapFault`).
    let mut host = Host::new();
    run_with_host(m, func, args, fuel, &mut host)
}

/// Like [`run`], but with a caller-provided [`Host`] (the powerbox): grant the entry
/// function's capabilities into `host`, pass their handle indices in `args`, then read
/// effects (`host.stdout`, etc.) back afterwards. This is how a capability-using guest
/// is driven (§3c/§3e).
pub fn run_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    if m.funcs.get(func as usize).is_none() {
        return Err(Trap::Malformed);
    }
    // One linear-memory window per run, zero-initialized and lazily paged. The whole module
    // shares it. The window is a large reserved range (§4 default policy) with only `mapped`
    // backed, so an out-of-`mapped` access faults (detect-and-kill) instead of wrapping.
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data); // §3a/D40 data segments (copy + RO-protect)
        mm
    });
    drive(&m.funcs, func, args, fuel, &mut mem, host)
}

/// Run the entry vCPU on the M:N executor: submit the root, become a worker on the calling thread,
/// and once every vCPU has finished, join any worker threads the executor spawned and read the root's
/// outcome back. `funcs` is cloned into an `Arc<[Func]>` the vCPUs own, so a spawned vCPU borrows
/// nothing and can run on a pooled thread. A single-threaded guest never spawns a worker — the calling
/// thread runs it to completion — so non-threaded runs pay no pool overhead.
fn drive(
    funcs: &[Func],
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    let funcs: Arc<[Func]> = funcs.to_vec().into();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, MAX_WORKERS);
    let sched = Arc::new(Scheduler::new(MAX_VCPUS, workers));
    // The powerbox is **shared** by every vCPU of the run (so spawned threads inherit it): move the
    // caller's host into an `Arc<Mutex<Host>>`, hand a clone to the root (and, on `thread.spawn`, to
    // each child), then unwrap it back into the caller after every vCPU is gone. The root still owns
    // the run's `mem`/`fuel`, read back from its outcome.
    let host_shared = Arc::new(Mutex::new(std::mem::take(host)));
    // §9/§12 async ring: wire the completion `notify` hook to this run's M:N scheduler, so an offload
    // worker waking a vCPU parked in `wait` is a `Scheduler::notify` on the confined counter key.
    {
        let sched_for_notify = Arc::clone(&sched);
        host_shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_async_notify(Arc::new(move |key, count| {
                sched_for_notify.notify(key, count);
            }));
    }
    let root_id = {
        let mut s = sched.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.workers = 1; // the calling thread acts as worker 0
        let root = Box::new(VCpu::new(
            funcs,
            entry,
            args,
            mem.take(),
            Arc::clone(&host_shared),
            *fuel,
            0,
            id,
            SchedRef::Real(Arc::clone(&sched)),
        ));
        s.runnable.push_back(root);
        id
    };
    // Run as worker 0 until the run shuts down (every vCPU finished), then join spawned workers.
    worker_loop(&sched);
    let handles = std::mem::take(&mut sched.lock().handles);
    for h in handles {
        let _ = h.join();
    }
    // Drain any in-flight async-ring offload jobs (they hold the window's `Arc<Region>` and may still
    // be writing the futex counter) and drop the `notify` hook's `Arc<Scheduler>` before reading the
    // final window back. Safe to lock: all vCPUs are gone, so the shared host is otherwise idle.
    {
        let mut h = host_shared.lock().unwrap_or_else(|e| e.into_inner());
        h.quiesce_pool();
        h.clear_async_notify();
    }
    let out = sched
        .lock()
        .results
        .remove(&root_id)
        .expect("root vCPU finished");
    *fuel = out.fuel;
    *mem = out.mem;
    // Every vCPU (which held an Arc clone) is finished and dropped now, so the shared host is uniquely
    // owned — unwrap it back into the caller so it observes the run's effects (stdout, grants, clock…).
    *host = Arc::try_unwrap(host_shared)
        .unwrap_or_else(|_| unreachable!("all vCPUs dropped before host readback"))
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    out.result
}

/// Like [`run`], but seed the window with `init_mem` (its low bytes) and return the final
/// window contents (the same number of bytes) alongside the result. This is the
/// **escape-oracle** path (§18): a *verified* module must keep every access in-window, so a
/// run that completes without trapping must leave a window byte-identical to the JIT's. The
/// non-zero seed makes a divergent (e.g. under-masked) *read* observable, not just a write.
/// With no declared memory the snapshot is empty.
pub fn run_capture(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed.
    run_capture_reserved(m, func, args, fuel, init_mem, DEFAULT_RESERVED_LOG2)
}

/// Like [`run_capture`], but with a host **reservation policy**: confinement masks into
/// `[0, 2^reserved_log2)` while only the declared `1 << size_log2` bytes are backed, so an
/// access into the reserved-but-unmapped tail faults (`Trap::MemoryFault`) instead of wrapping
/// (the deliberate I1 change for the §4 "guard-when-bounded" model). `reserved_log2` is raised
/// to at least `size_log2` (so `0` ⇒ fully mapped). This is the interpreter side of the
/// escape-oracle under the decoupled model and must be driven with the **same** `reserved_log2`
/// as the JIT's [`svm_jit::compile_and_run_capture_reserved`] to stay in differential lockstep.
pub fn run_capture_reserved(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data); // §3a/D40 data segments (after the escape-oracle seed)
        mm
    });
    let r = drive(&m.funcs, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    (r, snap)
}

/// Like [`run_capture_reserved`], but with a caller-provided [`Host`] (the powerbox), so a
/// `cap.call` to a *granted* handle takes its **success** path while the final-window snapshot
/// still feeds the escape-oracle (§18). Pairs with the JIT's
/// [`svm_jit::compile_and_run_capture_reserved_with_host`]: running both lets the §3e Memory
/// capability's `map`/`unmap`/`protect` effects be byte-compared across backends, not just their
/// return values — a real generative escape-oracle for the capability path.
/// Escape-oracle snapshot span (the `_with_host` capture): byte-compare the low `SNAP_CAP` bytes of
/// the window — *including* reserved-tail pages the guest grew via the Memory cap, not just the
/// backed prefix. **Must match `svm_jit`'s `SNAP_CAP`** so both backends snapshot the same span.
const SNAP_CAP: usize = 1 << 18; // 256 KiB

pub fn run_capture_reserved_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm
    });
    let r = drive(&m.funcs, func, args, fuel, &mut mem, host);
    // Snapshot past the backed prefix to also cover reserved-tail pages the guest grew (the §1a
    // growth path), matching the JIT's `_with_host` capture span so the escape-oracle byte-compares
    // them too.
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_window(SNAP_CAP))
        .unwrap_or_default();
    (r, snap)
}

/// Run the guest confined to a §14 **nested sub-window** `[base, base+size)` of a fully-backed
/// parent of `parent_bytes` (the child runs over the parent's `Region`; `size = 1 << size_log2` is
/// the module's declared memory). The masking unit ([`svm_mask::Window::sub`]) confines every child
/// access into its slice, so a *verified* guest reaches only `[base, base+size)`. This is the
/// interpreter side of the **sub-window escape-oracle**: pair it with the JIT's
/// [`svm_jit::compile_and_run_capture_sub`] and byte-compare the whole parent — every byte outside
/// the slice must stay as seeded (confinement) and the slice must match the JIT (codegen). `init_mem`
/// seeds the whole parent; the returned `Vec` is the whole parent window (`parent_bytes` bytes).
pub fn run_capture_sub(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    base: u64,
    parent_bytes: u64,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::sub_window(base, mc.size_log2, parent_bytes);
        mm.seed_parent(init_mem); // seed the whole parent, not just the child slice
        mm.init_data_at(&m.data, base); // child-relative segments shifted into the slice
        mm
    });
    let r = drive(&m.funcs, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_parent(parent_bytes))
        .unwrap_or_default();
    (r, snap)
}

/// Run a module under the **deterministic explorer** (§18) with scheduling decisions driven by
/// `seed`: a single OS thread interleaves the guest's vCPUs (green threads) cooperatively, so the run
/// is fully reproducible and sweeping seeds enumerates distinct interleavings. This is the
/// verification driver for concurrent guest code — no wall-clock, no OS-scheduler nondeterminism, so
/// a failing interleaving is replayable from its seed. Returns the entry vCPU's result (or the trap /
/// `ThreadFault` on a guest deadlock). Memory is default-reserved + data-initialized; no powerbox.
pub fn run_scheduled(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    seed: u64,
) -> Result<Vec<Value>, Trap> {
    if m.funcs.get(func as usize).is_none() {
        return Err(Trap::Malformed);
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();
    let mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data);
        mm
    });
    let det = Arc::new(DetSched::new(seed, MAX_VCPUS));
    let root_id = {
        let mut s = det.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        let root = Box::new(VCpu::new(
            funcs,
            func,
            args,
            mem,
            Arc::new(Mutex::new(Host::new())),
            fuel,
            0,
            id,
            SchedRef::Det(Arc::clone(&det)),
        ));
        s.runnable.push(root);
        id
    };
    run_det(&det);
    let out = det.lock().results.remove(&root_id);
    match out {
        Some(out) => out.result,
        None => Err(Trap::ThreadFault), // could not complete (a guest join-deadlock)
    }
}

/// Drives one run's scheduling choices for the exhaustive model checker, and records enough to walk
/// the next branch. `plan` is the choice to make at each decision point (a runnable set with >1
/// member); points past its end default to choice 0. `branches`/`chosen` log this run's actual
/// fan-out and choices so the caller can backtrack.
struct Choices {
    plan: Vec<usize>,
    branches: Vec<usize>,
    chosen: Vec<usize>,
    depth: usize,
}

impl Choices {
    fn new(plan: Vec<usize>) -> Choices {
        Choices {
            plan,
            branches: Vec::new(),
            chosen: Vec::new(),
            depth: 0,
        }
    }

    /// Pick a runnable index given `n` choices. A singleton runnable set is not a real decision (and
    /// isn't recorded), so the plan stays compact and stable across replays.
    fn pick(&mut self, n: usize) -> usize {
        if n == 1 {
            return 0;
        }
        let c = self.plan.get(self.depth).copied().unwrap_or(0).min(n - 1);
        self.branches.push(n);
        self.chosen.push(c);
        self.depth += 1;
        c
    }
}

/// The result of exhaustively exploring a concurrent program's interleavings ([`explore_all`]).
#[derive(Debug)]
pub struct Exhaustive {
    /// Every **distinct** terminal outcome observed across all explored schedules. For a correct
    /// program with an interleaving-invariant result this is a single element.
    pub outcomes: Vec<Result<Vec<Value>, Trap>>,
    /// How many complete schedules were run.
    pub schedules: u64,
    /// `true` if the whole interleaving tree was enumerated; `false` if `max_schedules` cut it short
    /// (so `outcomes` is a sound under-approximation, not a proof over *all* interleavings).
    pub complete: bool,
}

/// The shared object a visible op touches, used by [`explore_all`]'s DPOR to decide which
/// transitions **commute** (independent ⇒ their order is irrelevant) vs. **conflict** (dependent ⇒
/// both orders must be explored). Memory/atomic accesses are a confined byte range + read/write;
/// futex `wait`/`notify` are modelled as a read/write of their (confined, in-window) key, so they
/// also conflict with atomic accesses to the same word. `thread.spawn`/`join` carry no racy object —
/// their ordering is already enforced by the scheduler's *enabled* set (a child isn't runnable before
/// its spawn; a joiner/waiter is parked, not a scheduling choice).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MemAccess {
    /// No shared object (pure control/sync whose order the enabled set already fixes).
    None,
    /// A `[base, base+width)` byte range; `write` is true for any store/RMW/notify.
    Range { base: u64, width: u32, write: bool },
}

impl MemAccess {
    /// Two transitions are **dependent** (don't commute) iff they touch overlapping bytes and at
    /// least one writes — the standard read/write conflict relation DPOR reduces over.
    fn conflicts(self, other: MemAccess) -> bool {
        match (self, other) {
            (
                MemAccess::Range {
                    base: a,
                    width: wa,
                    write: wwa,
                },
                MemAccess::Range {
                    base: b,
                    width: wb,
                    write: wwb,
                },
            ) => (wwa || wwb) && a < b.saturating_add(wb as u64) && b < a.saturating_add(wa as u64),
            _ => false,
        }
    }
}

/// The confined object a visible instruction will access, computed from the live SSA values at the
/// decision point (mirrors what `load`/`store`/`atomic_*`/`prepare_wait` confine to). A confinement
/// failure (out-of-reserved) ⇒ [`MemAccess::None`]: the op will trap and the thread ends, contributing
/// no ordering constraint.
fn access_of(inst: &Inst, vals: &[Value], mem: &Option<Mem>) -> MemAccess {
    let Some(m) = mem.as_ref() else {
        return MemAccess::None;
    };
    let range = |addr: ValIdx, offset: u64, width: u32, write: bool| -> MemAccess {
        match get(vals, addr).and_then(as_i64) {
            Ok(a) => match m.confine_checked(a as u64, offset, width) {
                Ok(base) => MemAccess::Range { base, width, write },
                Err(_) => MemAccess::None,
            },
            Err(_) => MemAccess::None,
        }
    };
    match inst {
        Inst::Load {
            op, addr, offset, ..
        } => range(*addr, *offset, op.info().2, false),
        Inst::Store {
            op, addr, offset, ..
        } => range(*addr, *offset, op.info().2, true),
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), false),
        Inst::AtomicStore {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        Inst::AtomicRmw {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        Inst::AtomicCmpxchg {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        // Futex key: `wait` reads it (compared under the lock), `notify` writes it (the wake). Width 4
        // is the common i32 futex; an overlapping i64 atomic at the same word still conflicts by range.
        Inst::MemoryWait { ty, addr, .. } => range(*addr, 0, atomic_width(*ty), false),
        Inst::MemoryNotify { addr, .. } => range(*addr, 0, 4, true),
        _ => MemAccess::None, // ThreadSpawn / ThreadJoin: ordering via the enabled set, not a race
    }
}

/// One executed transition in a schedule: which vCPU ran (`tid`), the runnable vCPUs at that decision
/// (`enabled`, sorted — the choices that were available), and the object it touched (`access`). The
/// trace of these is what DPOR analyses for races.
struct SchedEvent {
    tid: TaskId,
    enabled: Vec<TaskId>,
    access: MemAccess,
}

/// Drives one schedule under DPOR with **sleep sets** (Flanagan–Godefroid). Follow `plan` (one
/// `TaskId` per decision); past its end pick the smallest enabled `TaskId` that is **not asleep**. As
/// it descends it carries the current `sleep` set — threads whose exploration from here would be
/// redundant: a thread sleeps in a subtree once a sibling *independent* with it has been explored, and
/// wakes the moment a *conflicting* transition runs. `prior[d]` supplies the siblings already explored
/// at depth `d` (with their accessed objects), so the inherited sleep is reconstructed during replay.
/// Records the executed `trace` for race analysis; sets `blocked` when every enabled vCPU is asleep
/// (a redundant prefix whose completions were covered by other schedules — the run stops, contributing
/// no outcome).
struct Dpor {
    plan: Vec<TaskId>,
    prior: Vec<BTreeMap<TaskId, MemAccess>>,
    depth: usize,
    sleep: BTreeMap<TaskId, MemAccess>,
    trace: Vec<SchedEvent>,
    pending: Option<(TaskId, Vec<TaskId>)>,
    blocked: bool,
}

impl Dpor {
    fn new(plan: Vec<TaskId>, prior: Vec<BTreeMap<TaskId, MemAccess>>) -> Dpor {
        Dpor {
            plan,
            prior,
            depth: 0,
            sleep: BTreeMap::new(),
            trace: Vec::new(),
            pending: None,
            blocked: false,
        }
    }

    /// Choose the next vCPU by `TaskId` (not runnable index — the runnable order is reshuffled by
    /// `swap_remove`, so addressing by id keeps the plan stable across replays). Returns `None` when
    /// every enabled vCPU is asleep, so the caller stops this (redundant) run. `enabled` is sorted.
    fn pick(&mut self, enabled: &[TaskId]) -> Option<TaskId> {
        let tid = if self.depth < self.plan.len() {
            // Forced replay (incl. a race-woken thread): the planned choice overrides the sleep set.
            self.plan[self.depth]
        } else {
            // Greedy extension: the smallest enabled thread that is not asleep here.
            match enabled
                .iter()
                .copied()
                .find(|t| !self.sleep.contains_key(t))
            {
                Some(t) => t,
                None => {
                    self.blocked = true;
                    return None;
                }
            }
        };
        debug_assert!(enabled.contains(&tid), "planned tid must be runnable");
        self.pending = Some((tid, enabled.to_vec()));
        Some(tid)
    }

    /// Finalize the current decision into the trace once its `access` is known, and advance the sleep
    /// set to the child state: the thread that just ran leaves the set (its old next-transition entry is
    /// stale); the siblings explored before it (`prior[depth]`) join it; then everything that
    /// **conflicts** with the transition just taken wakes (is dropped), leaving only the independent
    /// threads asleep deeper — the FG sleep-set rule `sleep(s.p) = {q ∈ sleep(s) : indep(p, q)}`.
    fn finish(&mut self, access: MemAccess) {
        if let Some((tid, enabled)) = self.pending.take() {
            self.trace.push(SchedEvent {
                tid,
                enabled,
                access,
            });
            self.sleep.remove(&tid);
            if let Some(prior) = self.prior.get(self.depth) {
                for (&q, &qacc) in prior {
                    self.sleep.entry(q).or_insert(qacc);
                }
            }
            self.sleep.retain(|_, &mut qacc| !access.conflicts(qacc));
            self.depth += 1;
        }
    }
}

/// One node of the DPOR exploration along the current depth-first path: the vCPU `chosen` here (and the
/// object `chosen_acc` it touched), the `enabled` set, the `backtrack`/`done` sets (threads still to
/// explore vs. already explored from this state), each explored thread's access (`done_acc`), and
/// `prior_acc` — the siblings explored *before* the current `chosen`, which seed the child sleep set
/// during replay. The Flanagan–Godefroid bookkeeping, plus the access maps sleep sets need.
struct DporSlot {
    chosen: TaskId,
    chosen_acc: MemAccess,
    enabled: Vec<TaskId>,
    backtrack: BTreeSet<TaskId>,
    done: BTreeSet<TaskId>,
    done_acc: BTreeMap<TaskId, MemAccess>,
    prior_acc: BTreeMap<TaskId, MemAccess>,
}

/// **Exhaustive interleaving model checker** (§18) with **dynamic partial-order reduction** (DPOR):
/// enumerate every distinct schedule of a concurrent guest *modulo independent-operation reordering*
/// and report the set of terminal outcomes — turning "sweep random seeds and hope" into a proof, for
/// programs small enough to explore fully.
///
/// It's a *stateless* checker (CHESS / `shuttle`-style): each schedule is one fresh execution replayed
/// from a planned sequence of scheduling choices, with no VM-state snapshotting. vCPUs run at
/// **memory-op granularity** (`memop` + `quantum = 1`), so the decision points are exactly the
/// shared-state / sync operations ([`is_visible`]). DPOR (Flanagan–Godefroid stateless form, **with
/// sleep sets**) then only explores *both* orders of two transitions when they actually **conflict**
/// ([`MemAccess::conflicts`]: same bytes, one a write); independent operations keep one order. After
/// each run it detects races (for each transition, the latest earlier conflicting transition by a
/// *different* vCPU) and adds the conflicting vCPU to that earlier decision's `backtrack` set; **sleep
/// sets** then prune the residual redundancy (a thread that became redundant after an independent
/// sibling ran is held asleep down that subtree until a conflict wakes it), so the search visits
/// essentially one schedule per Mazurkiewicz trace. It DFS-backtracks to the deepest decision with an
/// unexplored, non-sleeping alternative — stopping when the tree is exhausted or `max_schedules` is hit.
/// The reduction is sound: reordering independent ops cannot change the terminal state, so the set of
/// reachable outcomes is identical to the unreduced enumeration ([`explore_all_bruteforce`], the
/// differential oracle) at a fraction of the schedules.
///
/// Asserting `outcomes == [expected]` (with `complete`) proves the invariant holds under every
/// interleaving.
pub fn explore_all(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
) -> Exhaustive {
    if m.funcs.get(func as usize).is_none() {
        return Exhaustive {
            outcomes: vec![Err(Trap::Malformed)],
            schedules: 1,
            complete: true,
        };
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();

    let mut stack: Vec<DporSlot> = Vec::new();
    let mut outcomes: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    let mut schedules = 0u64;
    let complete;

    loop {
        // Replay the current path (`chosen` per depth), extending with the default choice past its end.
        // `prior` carries each slot's pre-`chosen` siblings so the controller reconstructs sleep sets.
        let plan: Vec<TaskId> = stack.iter().map(|s| s.chosen).collect();
        let prior: Vec<BTreeMap<TaskId, MemAccess>> =
            stack.iter().map(|s| s.prior_acc.clone()).collect();
        let mut dpor = Dpor::new(plan, prior);
        let result = run_one_schedule(
            &funcs,
            &m.memory,
            &m.data,
            func,
            args,
            fuel,
            Policy::Dpor(&mut dpor),
        );
        schedules += 1;
        // A sleep-blocked run is a redundant prefix (its completions were reached elsewhere): keep its
        // trace for race/backtrack bookkeeping, but don't record an outcome from its truncated tail.
        if !dpor.blocked && !outcomes.contains(&result) {
            outcomes.push(result);
        }
        let trace = dpor.trace;

        // Sync the path stack with the trace just produced. Slots for the replayed prefix already
        // exist (their `chosen` matches by construction); push fresh slots for newly reached depths;
        // a forced/blocked choice that shortened the run drops the now-stale deeper slots.
        if trace.len() < stack.len() {
            stack.truncate(trace.len());
        }
        for (d, ev) in trace.iter().enumerate() {
            if d < stack.len() {
                debug_assert_eq!(stack[d].chosen, ev.tid);
                stack[d].enabled.clone_from(&ev.enabled);
                stack[d].chosen_acc = ev.access;
                stack[d].done.insert(ev.tid);
                stack[d].done_acc.insert(ev.tid, ev.access);
            } else {
                stack.push(DporSlot {
                    chosen: ev.tid,
                    chosen_acc: ev.access,
                    enabled: ev.enabled.clone(),
                    backtrack: BTreeSet::from([ev.tid]),
                    done: BTreeSet::from([ev.tid]),
                    done_acc: BTreeMap::from([(ev.tid, ev.access)]),
                    prior_acc: BTreeMap::new(),
                });
            }
        }

        // Race detection (Flanagan–Godefroid): for each transition `j`, find the latest earlier
        // transition `i` by a *different* vCPU that conflicts with it, and ensure the decision at `i`
        // will also try `j`'s vCPU (or, if it wasn't co-enabled there, every enabled vCPU — the
        // conservative "may-be-co-enabled" fallback). The recursion across runs then covers earlier
        // conflicts. A race-added thread overrides the sleep set (backtrack ∖ done isn't pruned by it).
        for j in 0..trace.len() {
            for i in (0..j).rev() {
                if trace[i].tid != trace[j].tid && trace[i].access.conflicts(trace[j].access) {
                    let q = trace[j].tid;
                    if stack[i].enabled.contains(&q) {
                        stack[i].backtrack.insert(q);
                    } else {
                        let enabled = stack[i].enabled.clone();
                        stack[i].backtrack.extend(enabled);
                    }
                    break;
                }
            }
        }

        // Backtrack to the deepest decision with an unexplored alternative; force it next run, recording
        // the now-explored siblings as its child's sleep seed (`prior_acc`).
        let mut next = None;
        for d in (0..stack.len()).rev() {
            if let Some(&p) = stack[d].backtrack.difference(&stack[d].done).next() {
                next = Some((d, p));
                break;
            }
        }
        match next {
            Some((d, p)) if schedules < max_schedules => {
                stack[d].prior_acc = stack[d].done_acc.clone();
                stack[d].done.insert(p);
                stack[d].chosen = p;
                stack.truncate(d + 1);
            }
            Some(_) => {
                complete = false;
                break;
            }
            None => {
                complete = true;
                break;
            }
        }
    }

    Exhaustive {
        outcomes,
        schedules,
        complete,
    }
}

/// The **unreduced** exhaustive enumerator — explores *every* ordering of visible ops, including
/// reorderings of independent operations. Superseded by [`explore_all`] (DPOR) for real use; kept as
/// the differential oracle that proves DPOR's reduction is sound (same `outcomes`, fewer `schedules`).
#[doc(hidden)]
pub fn explore_all_bruteforce(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
) -> Exhaustive {
    if m.funcs.get(func as usize).is_none() {
        return Exhaustive {
            outcomes: vec![Err(Trap::Malformed)],
            schedules: 1,
            complete: true,
        };
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();

    let mut plan: Vec<usize> = Vec::new();
    let mut outcomes: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    let mut schedules = 0u64;
    let complete;

    loop {
        let mut choices = Choices::new(plan);
        let result = run_one_schedule(
            &funcs,
            &m.memory,
            &m.data,
            func,
            args,
            fuel,
            Policy::Brute(&mut choices),
        );
        schedules += 1;
        if !outcomes.contains(&result) {
            outcomes.push(result);
        }

        // Backtrack: bump the deepest decision that has an unexplored sibling, dropping everything
        // after it (those subtrees are re-explored fresh under the new prefix).
        let mut next = None;
        for i in (0..choices.branches.len()).rev() {
            if choices.chosen[i] + 1 < choices.branches[i] {
                let mut p = choices.chosen[..i].to_vec();
                p.push(choices.chosen[i] + 1);
                next = Some(p);
                break;
            }
        }
        match next {
            Some(p) if schedules < max_schedules => plan = p,
            Some(_) => {
                complete = false;
                break;
            }
            None => {
                complete = true;
                break;
            }
        }
    }

    Exhaustive {
        outcomes,
        schedules,
        complete,
    }
}

/// Run a single schedule under the exhaustive checker: a fresh memory image and root vCPU (at
/// memory-op granularity), driven by `policy` ([`Policy::Brute`] or [`Policy::Dpor`]). Returns the root
/// task's outcome.
fn run_one_schedule(
    funcs: &Arc<[Func]>,
    memory: &Option<Memory>,
    data: &[Data],
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    policy: Policy,
) -> Result<Vec<Value>, Trap> {
    let mem = memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(data);
        mm
    });
    let det = Arc::new(DetSched::new(0, MAX_VCPUS)); // seed unused under the exhaustive policy
    let root_id = {
        let mut s = det.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        let mut root = VCpu::new(
            Arc::clone(funcs),
            func,
            args,
            mem,
            Arc::new(Mutex::new(Host::new())),
            fuel,
            0,
            id,
            SchedRef::Det(Arc::clone(&det)),
        );
        root.memop = true;
        s.runnable.push(Box::new(root));
        id
    };
    run_with_policy(&det, policy);
    let out = det.lock().results.remove(&root_id);
    out.map_or(Err(Trap::ThreadFault), |o| o.result)
}

/// One activation record on the **explicit** guest call stack (§12). Reifying the call
/// stack — rather than recursing on the host stack — is what makes fibers possible: a
/// fiber's continuation is exactly its `Vec<Frame>`, which `suspend` pauses and
/// `cont.resume` restarts.
struct Frame {
    /// The function this activation is executing — stored as an **index** (not a borrow) so a
    /// `Frame` (hence a whole vCPU continuation) is self-contained and movable between worker
    /// threads. Resolved against the vCPU's owned `Arc<[Func]>` at each use.
    func: FuncIdx,
    /// Index of the block currently executing.
    block: usize,
    /// Index of the **next** instruction to execute within that block. Saved across a
    /// nested call so the caller resumes just past the `call` when the callee returns.
    inst: usize,
    /// Block-local SSA values produced so far (entry = the call arguments).
    vals: Vec<Value>,
}

/// Maximum number of fibers a single run may create (§12). Bounds the fiber table so a
/// fiber-bomb yields a clean [`Trap::FiberFault`] instead of unbounded host allocation —
/// the reference-oracle analogue of the quota that charges out-of-band stacks to the
/// guest, so a fiber-bomb OOMs *itself*, never the host.
const MAX_FIBERS: usize = 1 << 16;

/// Maximum number of **concurrently live** vCPUs (`thread.spawn`) across a run (§12). With the M:N
/// executor a vCPU is a cheap green thread (a parked one costs only its continuation, not an OS
/// thread), so this can be large; it's just an anti-bomb ceiling — exceeding it is a clean
/// [`Trap::ThreadFault`]. A spawned-and-joined loop creates unboundedly many vCPUs over its lifetime;
/// only simultaneous liveness is bounded.
const MAX_VCPUS: usize = 1 << 16;

/// `cont.resume` status results (§12): the fiber `suspend`ed (resumable) vs. returned (done).
const FIBER_SUSPENDED: i32 = 0;
const FIBER_RETURNED: i32 = 1;
/// Extra §14 coroutine-`resume` status: the child suspended on a **page fault** (its `(status, value)`
/// is `(2, fault_addr)`) — the parent supplies the page and resumes (fault-driven yield / lazy paging).
const CORO_FAULTED: i32 = 2;

/// `<ty>.atomic.wait` status results (§12), matching wasm: woken by a notify / value mismatch / timed
/// out.
const WAIT_WOKEN: i32 = 0;
const WAIT_NOT_EQUAL: i32 = 1;
const WAIT_TIMED_OUT: i32 = 2;

/// Upper bound on how long a `<ty>.atomic.wait` will actually block, regardless of the guest's
/// requested timeout (and what a negative — "infinite" — timeout is clamped to). A vCPU blocking
/// forever would never let the run's thread `scope` join; capping keeps the host live (a guest can
/// stall *itself* but not wedge the process). Legitimate waits return immediately on the notify, so
/// the cap only bounds the missed-notify fallback.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Maximum worker OS threads the executor will spawn for one run (the "N" of M:N). Capped at the
/// host parallelism. Workers are spawned **lazily** — a single-threaded guest never creates any.
const MAX_WORKERS: usize = 32;

/// A task identifier (a spawned vCPU). Distinct from the per-vCPU join *handle* (an index into the
/// spawner's child table); the executor keys results/waiters by `TaskId`.
type TaskId = u64;

/// A finished vCPU's outcome, parked in the scheduler until a `thread.join` claims it (or, for the
/// root, until [`drive`] reads it). Carries `mem`/`fuel` so the root's window can be snapshot and its
/// fuel read back after the worker that ran it is gone. (The powerbox is **shared** across all vCPUs of
/// the run — `Arc<Mutex<Host>>` — so it isn't carried here; `drive` reads it back by unwrapping the Arc.)
struct Outcome {
    result: Result<Vec<Value>, Trap>,
    mem: Option<Mem>,
    fuel: u64,
}

/// Why a vCPU yielded its worker (returned by [`VCpu::run`]).
enum Blocked {
    /// Blocked in `thread.join` on child task `child` (the join handle's table slot is recorded in the
    /// vCPU's `pending`, set before it parks).
    Join { child: TaskId },
    /// Blocked in `atomic.wait` on confined address `key`, to wake on a matching `notify` or after
    /// `timeout_ns` (already `MAX_WAIT`-clamped). `expected`/`width` let the driver re-check the value
    /// under its lock (the futex compare-and-park atomicity). The driver turns `timeout_ns` into a
    /// deadline on *its* clock — wall-clock for the real pool, a logical clock for the explorer.
    Wait {
        key: u64,
        expected: u64,
        width: u32,
        timeout_ns: u64,
    },
}

/// Set on a parked vCPU before it is re-enqueued, telling its driver how to finish the op on resume.
enum Pending {
    /// Finish a `thread.join`: take the child's result from `threads[slot]`.
    Join { slot: usize },
    /// Finish an `atomic.wait`, pushing this status (woken / not-equal / timed-out).
    Wait(i32),
    /// Finish a §14 co-fiber `yield`: push the value the parent's `resume` delivered (the result of
    /// the child's `Yielder` cap.call). Only ever set on a *coroutine* child the parent drives inline.
    CoResume(i64),
}

/// One run of a vCPU until it finishes or yields.
enum Step {
    Done(Result<Vec<Value>, Trap>),
    Park(Blocked),
    /// Ran out its scheduling quantum mid-execution (deterministic-explorer preemption); re-enqueue
    /// and continue later. The real executor uses an unbounded quantum and never yields.
    Yield,
}

/// Internal `?`-friendly driver result; [`VCpu::run`] folds an `Err` into `Step::Done(Err)`.
enum Inner {
    Done(Vec<Value>),
    Park(Blocked),
    Yield,
    /// A §14 **co-fiber** child yielded a value to its instantiator-parent (`Yielder` cap.call). The
    /// child's continuation (frames/mem/host) is preserved in the `VCpu` so the parent's next `resume`
    /// continues it. Only produced while a coroutine child is driven inline by `resume`; a normal vCPU
    /// that reaches it (a `Yielder` with no resumer) is a `FiberFault`.
    CoYield(i64),
    /// A §14 **fault-driven yield**: a coroutine child (`fault_yields`) hit a recoverable page fault
    /// (an access to an unmapped page in its window) at this confined address. The faulting access has
    /// been rewound; the parent's `resume` supplies the page and re-runs it (userfaultfd-style lazy
    /// paging). Like `CoYield`, only produced for an inline-driven coroutine.
    CoFault(u64),
}

/// The **M:N executor** (§12): a bounded pool of worker OS threads runs many vCPUs (green threads)
/// from a shared run-queue. A vCPU that blocks on `thread.join`/`atomic.wait` **parks** — its owned
/// continuation ([`VCpu`]) is set aside, freeing the worker — and is re-enqueued when the awaited
/// event fires (child completion / `notify` / timeout). Thus thousands of vCPUs run on a handful of
/// threads. One mutex guards all scheduler state: coarse, but obviously race-free (the interpreter is
/// the reference oracle; the JIT is the performance path). Workers are spawned lazily, so a
/// single-threaded guest runs entirely on the calling thread with no pool at all.
struct Scheduler {
    mx: Mutex<Sched>,
    /// Workers wait here for runnable vCPUs (woken on new work, shutdown, or a timer deadline).
    /// `drive` runs as worker 0 and returns when shutdown fires, so no separate idle signal is needed.
    work: Condvar,
    /// Max concurrently-live vCPUs (anti-bomb) and max worker threads.
    cap: usize,
    max_workers: usize,
}

#[derive(Default)]
struct Sched {
    /// vCPUs ready to run.
    runnable: VecDeque<Box<VCpu>>,
    /// Finished tasks' outcomes, awaiting `join` (or the root, awaiting `drive`).
    results: BTreeMap<TaskId, Outcome>,
    /// A vCPU parked in `join`, keyed by the child it awaits.
    join_waiters: BTreeMap<TaskId, Box<VCpu>>,
    /// vCPUs parked in `wait`, keyed by confined address; each tagged with a waiter id.
    wait_waiters: BTreeMap<u64, Vec<(u64, Box<VCpu>)>>,
    /// Min-heap of `(deadline, waiter id, address)` for timing out `wait`s.
    timers: BinaryHeap<Reverse<(Instant, u64, u64)>>,
    /// OS-thread handles of spawned workers (joined by `drive` at the end).
    handles: Vec<std::thread::JoinHandle<()>>,
    /// vCPUs not yet finished (running + queued + parked). The run ends when this hits 0.
    live: usize,
    /// Worker threads in existence (incl. the calling thread, counted as 1).
    workers: usize,
    next_task: TaskId,
    next_wid: u64,
    shutdown: bool,
}

impl Scheduler {
    fn new(cap: usize, max_workers: usize) -> Scheduler {
        Scheduler {
            mx: Mutex::new(Sched::default()),
            work: Condvar::new(),
            cap,
            max_workers,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Sched> {
        self.mx.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Allocate a task id + live slot and enqueue the vCPU built by `make`; spawn another worker if
    /// demand warrants. `None` if the live cap is hit (a thread-bomb).
    fn spawn(self: &Arc<Self>, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        let mut s = self.lock();
        if s.live >= self.cap {
            return None;
        }
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.runnable.push_back(make(id));
        self.maybe_spawn_worker(&mut s);
        self.work.notify_one();
        Some(id)
    }

    /// Grow the pool toward `min(live, max_workers)` so parked/queued vCPUs have a thread to run them.
    fn maybe_spawn_worker(self: &Arc<Self>, s: &mut Sched) {
        if s.workers < self.max_workers && s.workers < s.live && !s.shutdown {
            s.workers += 1;
            let me = Arc::clone(self);
            s.handles.push(std::thread::spawn(move || worker_loop(&me)));
        }
    }

    /// Wake up to `count` vCPUs parked on `key`; return how many were woken.
    fn notify(&self, key: u64, count: u32) -> u32 {
        let mut s = self.lock();
        let mut woken: Vec<Box<VCpu>> = Vec::new();
        if let Some(q) = s.wait_waiters.get_mut(&key) {
            while (woken.len() as u32) < count {
                match q.pop() {
                    Some((_, v)) => woken.push(v),
                    None => break,
                }
            }
            if q.is_empty() {
                s.wait_waiters.remove(&key);
            }
        }
        let n = woken.len() as u32;
        for mut v in woken {
            v.pending = Some(Pending::Wait(WAIT_WOKEN));
            s.runnable.push_back(v);
        }
        if n > 0 {
            self.work.notify_all();
        }
        n
    }
}

/// Move any expired `wait` timers' vCPUs back to the run-queue with a timed-out status. (A waiter
/// already woken by `notify` is simply absent — its stale timer is skipped.)
fn process_timers(s: &mut Sched) {
    let now = Instant::now();
    while let Some(&Reverse((dl, wid, key))) = s.timers.peek() {
        if dl > now {
            break;
        }
        s.timers.pop();
        let mut woken = None;
        if let Some(q) = s.wait_waiters.get_mut(&key) {
            if let Some(pos) = q.iter().position(|(id, _)| *id == wid) {
                woken = Some(q.remove(pos).1);
            }
        }
        if let Some(mut v) = woken {
            if s.wait_waiters.get(&key).is_some_and(|q| q.is_empty()) {
                s.wait_waiters.remove(&key);
            }
            v.pending = Some(Pending::Wait(WAIT_TIMED_OUT));
            s.runnable.push_back(v);
        }
    }
}

/// A worker: pull a runnable vCPU and dispatch it, sleeping (until work, a timer, or shutdown) when
/// idle. Returns when the run is shutting down and nothing is left to do.
fn worker_loop(sched: &Arc<Scheduler>) {
    loop {
        let next = {
            let mut s = sched.lock();
            loop {
                process_timers(&mut s);
                if let Some(v) = s.runnable.pop_front() {
                    break Some(v);
                }
                if s.shutdown {
                    break None;
                }
                match s.timers.peek().map(|Reverse((dl, _, _))| *dl) {
                    Some(dl) => {
                        let now = Instant::now();
                        if dl > now {
                            let (g, _) = sched
                                .work
                                .wait_timeout(s, dl - now)
                                .unwrap_or_else(|e| e.into_inner());
                            s = g;
                        }
                    }
                    None => s = sched.work.wait(s).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        match next {
            Some(v) => dispatch(sched, v),
            None => return,
        }
    }
}

/// Run one vCPU until it yields, then route the outcome: publish a result (waking a joiner) and
/// retire the slot, or park it on a join target / wait address.
fn dispatch(sched: &Arc<Scheduler>, mut v: Box<VCpu>) {
    match v.run(u64::MAX) {
        Step::Done(result) => {
            let id = v.id;
            let outcome = Outcome {
                result,
                mem: v.mem.take(),
                fuel: v.fuel,
            };
            drop(v);
            let mut s = sched.lock();
            if let Some(parent) = s.join_waiters.remove(&id) {
                s.runnable.push_back(parent);
                sched.work.notify_one();
            }
            s.results.insert(id, outcome);
            s.live -= 1;
            if s.live == 0 {
                s.shutdown = true;
                sched.work.notify_all();
            }
        }
        Step::Park(Blocked::Join { child }) => {
            let mut s = sched.lock();
            if s.results.contains_key(&child) {
                // Already finished between the join check and here — resume immediately.
                s.runnable.push_back(v);
                sched.work.notify_one();
            } else {
                s.join_waiters.insert(child, v);
            }
        }
        Step::Park(Blocked::Wait {
            key,
            expected,
            width,
            timeout_ns,
        }) => {
            let deadline = Instant::now() + Duration::from_nanos(timeout_ns);
            let mut s = sched.lock();
            // Re-read the value **under the lock** so the compare-and-park is atomic vs. `notify`.
            if v.atomic_value(key, width) != expected {
                v.pending = Some(Pending::Wait(WAIT_NOT_EQUAL));
                s.runnable.push_back(v);
                sched.work.notify_one();
            } else {
                let wid = s.next_wid;
                s.next_wid += 1;
                s.timers.push(Reverse((deadline, wid, key)));
                s.wait_waiters.entry(key).or_default().push((wid, v));
                sched.work.notify_all(); // let idle workers recompute their timer deadline
            }
        }
        Step::Yield => {
            // Unreachable for the real pool (quantum is `u64::MAX`), but re-enqueue for safety.
            let mut s = sched.lock();
            s.runnable.push_back(v);
            sched.work.notify_one();
        }
    }
}

/// A vCPU's executor handle: spawn/notify route to either the real OS-thread pool or the
/// single-threaded deterministic explorer. `Clone` is a cheap `Arc` bump (the child inherits it).
#[derive(Clone)]
enum SchedRef {
    Real(Arc<Scheduler>),
    Det(Arc<DetSched>),
}

impl SchedRef {
    fn spawn(&self, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        match self {
            SchedRef::Real(s) => s.spawn(make),
            SchedRef::Det(d) => d.spawn(make),
        }
    }
    fn notify(&self, key: u64, count: u32) -> u32 {
        match self {
            SchedRef::Real(s) => s.notify(key, count),
            SchedRef::Det(d) => d.notify(key, count),
        }
    }
    /// Take a finished child's outcome (for a resuming `thread.join`).
    fn take_result(&self, id: TaskId) -> Option<Outcome> {
        match self {
            SchedRef::Real(s) => s.lock().results.remove(&id),
            SchedRef::Det(d) => d.lock().results.remove(&id),
        }
    }
}

/// Upper bound on the deterministic explorer's per-step quantum (instructions before a forced
/// yield). A small bound interleaves vCPUs finely; the actual quantum each turn is seeded in
/// `1..=MAX_QUANTUM`, so varying the seed varies the interleaving.
const MAX_QUANTUM: u64 = 8;

/// The **deterministic explorer** (§18): a single-threaded, seed-driven executor for *verifying*
/// concurrent guest code. It runs the same vCPUs as the real pool but on one OS thread, choosing
/// which runnable vCPU to step (and for how long) from a seeded PRNG, and timing out `atomic.wait`s
/// on a **logical** clock. So a run is fully reproducible from its seed, and sweeping seeds explores
/// distinct interleavings — turning "run many times and hope" into systematic coverage, with any
/// failure replayable. No data races exist (one thread), so each seed realizes one valid sequential
/// interleaving of the shared-memory ops.
struct DetSched {
    st: Mutex<DetState>,
}

struct DetWaiter {
    key: u64,
    deadline: u64, // logical ns
    vcpu: Box<VCpu>,
}

/// A vCPU the explorer parked because it was **spinning**: it ran a visible op that changed no memory
/// and returned it to the same local configuration (a busy-wait retry). It stays parked — not a
/// scheduling choice, so the spin doesn't multiply the interleaving tree — until another vCPU writes to
/// the `[base, base+width)` range it was reading, which may have changed the value it spins on.
struct SpinWaiter {
    vcpu: Box<VCpu>,
    base: u64,
    width: u32,
}

struct DetState {
    // `Box` (matching the join/wait waiter maps) keeps moving a large vCPU between the runnable set
    // and the waiter collections a pointer copy.
    #[allow(clippy::vec_box)]
    runnable: Vec<Box<VCpu>>,
    results: BTreeMap<TaskId, Outcome>,
    join_waiters: BTreeMap<TaskId, Box<VCpu>>,
    wait_waiters: Vec<DetWaiter>,
    /// vCPUs parked by spin-loop detection (memop explorer only), woken by a write to their read range.
    spin_waiters: Vec<SpinWaiter>,
    live: usize,
    next_task: TaskId,
    clock: u64, // logical nanoseconds, advanced only to fire a timeout
    rng: u64,
    cap: usize,
}

impl DetState {
    /// Move every spin-parked vCPU whose read range `[base, base+width)` overlaps the just-written
    /// `[w_base, w_base+w_width)` back to the runnable set — a write there may have changed the value it
    /// spins on, so it must re-check (it re-parks if still stuck). The interpreter is sequentially
    /// consistent, so a write is the *only* way a spinner's read can change.
    fn wake_spins(&mut self, w_base: u64, w_width: u32) {
        let mut i = 0;
        while i < self.spin_waiters.len() {
            let s = &self.spin_waiters[i];
            let overlap = w_base < s.base.saturating_add(s.width as u64)
                && s.base < w_base.saturating_add(w_width as u64);
            if overlap {
                let w = self.spin_waiters.swap_remove(i);
                self.runnable.push(w.vcpu);
            } else {
                i += 1;
            }
        }
    }
}

impl DetState {
    /// xorshift64* — the seeded source of all scheduling choices (so the whole run is a function of
    /// the seed).
    fn rng(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

impl DetSched {
    fn new(seed: u64, cap: usize) -> DetSched {
        DetSched {
            st: Mutex::new(DetState {
                runnable: Vec::new(),
                results: BTreeMap::new(),
                join_waiters: BTreeMap::new(),
                wait_waiters: Vec::new(),
                spin_waiters: Vec::new(),
                live: 0,
                next_task: 0,
                clock: 0,
                rng: seed | 1,
                cap,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DetState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn spawn(&self, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        let mut s = self.lock();
        if s.live >= s.cap {
            return None;
        }
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.runnable.push(make(id));
        Some(id)
    }

    /// Wake up to `count` vCPUs waiting on `key`, in deterministic (insertion) order.
    fn notify(&self, key: u64, count: u32) -> u32 {
        let mut s = self.lock();
        let mut woken = 0u32;
        let mut i = 0;
        while woken < count && i < s.wait_waiters.len() {
            if s.wait_waiters[i].key == key {
                let mut w = s.wait_waiters.remove(i);
                w.vcpu.pending = Some(Pending::Wait(WAIT_WOKEN));
                s.runnable.push(w.vcpu);
                woken += 1;
            } else {
                i += 1;
            }
        }
        woken
    }
}

/// How a deterministic-explorer run resolves its two scheduling choices — which runnable vCPU to step
/// next, and for how long. `Seeded` is the random seed sweep ([`run_scheduled`]); `Exhaustive` is the
/// stateless model checker ([`explore_all`]), which follows a planned choice sequence and records the
/// branch factor at each point so the caller can DFS the whole interleaving tree.
enum Policy<'a> {
    Seeded,
    /// Unreduced enumeration ([`explore_all_bruteforce`]): pick by runnable *index* per `Choices`.
    Brute(&'a mut Choices),
    /// DPOR ([`explore_all`]): pick by `TaskId` per the plan, recording each step's [`MemAccess`].
    Dpor(&'a mut Dpor),
}

/// Run a deterministic-explorer instance to completion (every vCPU finished, or a genuine deadlock —
/// nothing runnable and nothing waiting on a timeout). Picks a runnable vCPU and a quantum per the
/// [`Policy`]; when none is runnable, advances the logical clock to the earliest `wait` deadline and
/// times that waiter out.
fn run_det(det: &Arc<DetSched>) {
    run_with_policy(det, Policy::Seeded);
}

fn run_with_policy(det: &Arc<DetSched>, mut policy: Policy) {
    loop {
        // Choose the next vCPU + quantum under the lock; release it before running (run_inner may
        // re-enter via `spawn`/`notify`).
        let (mut v, quantum) = {
            let mut s = det.lock();
            if s.runnable.is_empty() {
                if s.live == 0 {
                    return; // all done
                }
                // No one runnable: fire the earliest timeout (or deadlock if none).
                let Some(idx) =
                    (0..s.wait_waiters.len()).min_by_key(|&i| s.wait_waiters[i].deadline)
                else {
                    return; // live > 0 but nothing runnable/waiting: a guest join-deadlock
                };
                let mut w = s.wait_waiters.remove(idx);
                s.clock = s.clock.max(w.deadline);
                w.vcpu.pending = Some(Pending::Wait(WAIT_TIMED_OUT));
                s.runnable.push(w.vcpu);
                continue;
            }
            let n = s.runnable.len();
            // One visible op per turn (`memop` vCPUs) so every shared-state access is a decision.
            let (pick, quantum) = match &mut policy {
                Policy::Seeded => ((s.rng() as usize) % n, 1 + s.rng() % MAX_QUANTUM),
                Policy::Brute(c) => (c.pick(n), 1),
                Policy::Dpor(d) => {
                    // Address by `TaskId` (the runnable order is reshuffled by `swap_remove`), so the
                    // plan replays identically. `enabled` sorted ⇒ a stable default (smallest id). A
                    // `None` pick means every runnable vCPU is asleep ⇒ stop this redundant schedule.
                    let mut enabled: Vec<TaskId> = s.runnable.iter().map(|v| v.id).collect();
                    enabled.sort_unstable();
                    match d.pick(&enabled) {
                        Some(tid) => {
                            let idx = s
                                .runnable
                                .iter()
                                .position(|v| v.id == tid)
                                .expect("planned tid is runnable");
                            (idx, 1)
                        }
                        None => return,
                    }
                }
            };
            let v = s.runnable.swap_remove(pick);
            (v, quantum)
        };
        // Spin-loop detection (memop explorer only): snapshot the vCPU's local configuration and its
        // memory-write count so that, after the turn, a pure busy-wait retry (same config, no memory
        // changed) is distinguishable from real progress.
        let spin_capable = v.memop;
        let pre_fp = if spin_capable {
            v.local_fingerprint()
        } else {
            0
        };
        let writes_before = v.mem.as_ref().map_or(0, |m| m.writes);

        let step = v.run(quantum);

        let acc = v.acc.take();
        // DPOR: finalize this decision's trace entry now that the step's accessed object is known.
        if let Policy::Dpor(d) = &mut policy {
            d.finish(acc.unwrap_or(MemAccess::None));
        }
        // A turn that actually changed a byte may unblock spinners parked on that address — wake them
        // to re-check (they re-park if still stuck). Memory change is the only thing that can, under
        // sequential consistency, alter a parked spinner's read.
        let mem_changed = v.mem.as_ref().map_or(0, |m| m.writes) != writes_before;
        if mem_changed {
            if let Some(MemAccess::Range { base, width, .. }) = acc {
                det.lock().wake_spins(base, width);
            }
        }
        match step {
            Step::Done(result) => {
                let id = v.id;
                let outcome = Outcome {
                    result,
                    mem: v.mem.take(),
                    fuel: v.fuel,
                };
                drop(v);
                let mut s = det.lock();
                if let Some(parent) = s.join_waiters.remove(&id) {
                    s.runnable.push(parent);
                }
                s.results.insert(id, outcome);
                s.live -= 1;
            }
            Step::Park(Blocked::Join { child }) => {
                let mut s = det.lock();
                if s.results.contains_key(&child) {
                    s.runnable.push(v); // already done (pending already set)
                } else {
                    s.join_waiters.insert(child, v);
                }
            }
            Step::Park(Blocked::Wait {
                key,
                expected,
                width,
                timeout_ns,
            }) => {
                let mut s = det.lock();
                if v.atomic_value(key, width) != expected {
                    v.pending = Some(Pending::Wait(WAIT_NOT_EQUAL));
                    s.runnable.push(v);
                } else {
                    let deadline = s.clock.saturating_add(timeout_ns);
                    s.wait_waiters.push(DetWaiter {
                        key,
                        deadline,
                        vcpu: v,
                    });
                }
            }
            Step::Yield => {
                // Spin-park: the turn ran one visible op, changed no memory, and returned the vCPU to
                // the same local configuration — a busy-wait whose only way forward is another vCPU
                // writing what it just read. Park it off the runnable set (so the spin doesn't multiply
                // the interleaving tree, and an unfair "spin forever" schedule can't starve the writer)
                // until such a write wakes it. Anything else re-enqueues normally.
                if spin_capable && !mem_changed && v.local_fingerprint() == pre_fp {
                    if let Some(MemAccess::Range { base, width, .. }) = acc {
                        let mut s = det.lock();
                        s.spin_waiters.push(SpinWaiter {
                            vcpu: v,
                            base,
                            width,
                        });
                        continue;
                    }
                }
                det.lock().runnable.push(v);
            }
        }
    }
}

/// A §12 fiber: a first-class suspendable computation whose continuation is exactly its
/// reified call stack. `cont.new` makes one (`Pending`); `cont.resume` switches into it;
/// `suspend` switches back out, parking it (`Live`).
enum Fiber {
    /// Created by `cont.new`, not yet started: holds the `i32` funcref to launch on the
    /// first resume (resolved then through the function table as `(i64 sp, i64 arg) ->
    /// i64`) and the `i64` data-stack base `sp` to run it on.
    Pending { func: i32, sp: i64 },
    /// Started and currently **parked** — suspended at a `suspend`, or an ancestor in the
    /// resume chain — holding its reified call stack, ready to continue.
    Live(Vec<Frame>),
    /// The fiber whose frames are currently *in flight* in the driver's local `frames`.
    /// A placeholder keeping the table slot addressable (a handle resolving here is in the
    /// resume chain → inert / traps).
    Running,
    /// Returned: resuming it again traps.
    Done,
}

/// The fixed fiber entry signature (§12): a fiber runs a function of type `(i64 sp, i64
/// arg) -> i64`. `sp` is the fiber's data-stack base (the §3d two-stack split — every
/// frontend-emitted function already takes the data-SP as its first param); `arg` carries
/// the first-resume value in and the final value out (a window pointer can carry richer
/// payloads).
fn fiber_sig() -> FuncType {
    FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    }
}

/// Total live activation records across the resume chain — the running fiber's `frames`
/// (length `current_len`) plus every parked ancestor's stack. Bounds recursion across
/// *all* active fibers, not just the current one. With no fibers (`chain == [0]`) this is
/// just `current_len`, so the depth bound is unchanged from the single-stack case.
fn live_frames(fibers: &[Fiber], chain: &[usize], current_len: usize) -> usize {
    let mut total = current_len;
    for &i in &chain[..chain.len() - 1] {
        if let Fiber::Live(f) = &fibers[i] {
            total += f.len();
        }
    }
    total
}

/// Resolve a `cont.resume` handle to a table slot (§12). The handle is forgeable, so it is
/// **masked** into the power-of-two-padded table (Spectre-safe, like `call_indirect`), then
/// the slot is checked: in the resume chain (would alias a running stack), `Running`, or
/// `Done` ⇒ inert ([`Trap::FiberFault`]); only a `Pending`/`Live` slot is resumable.
fn resolve_fiber(fibers: &[Fiber], chain: &[usize], handle: i32) -> Result<usize, Trap> {
    let mask = fibers.len().next_power_of_two() - 1;
    let slot = (handle as u32 as usize) & mask;
    if slot >= fibers.len() || chain.contains(&slot) {
        return Err(Trap::FiberFault);
    }
    match &fibers[slot] {
        Fiber::Pending { .. } | Fiber::Live(_) => Ok(slot),
        Fiber::Running | Fiber::Done => Err(Trap::FiberFault),
    }
}

/// Resolve a `thread.join` handle to a table slot (§12). Like [`resolve_fiber`], the handle is
/// forgeable, so it is **masked** into the power-of-two-padded table, then bounds- and
/// liveness-checked: out of range or an already-joined (`None`) slot is inert ([`Trap::ThreadFault`]).
fn resolve_thread<T>(threads: &[Option<T>], handle: i32) -> Result<usize, Trap> {
    if threads.is_empty() {
        return Err(Trap::ThreadFault);
    }
    let mask = threads.len().next_power_of_two() - 1;
    let slot = (handle as u32 as usize) & mask;
    if slot >= threads.len() || threads[slot].is_none() {
        return Err(Trap::ThreadFault);
    }
    Ok(slot)
}

/// Take a parked (`Live`) fiber's frames out for execution, marking its slot `Running`.
fn take_running(fibers: &mut [Fiber], i: usize) -> Result<Vec<Frame>, Trap> {
    match std::mem::replace(&mut fibers[i], Fiber::Running) {
        Fiber::Live(f) => Ok(f),
        _ => Err(Trap::Malformed),
    }
}

/// Run one vCPU. `funcs` is an `Arc<[Func]>` the vCPU **owns** (a child gets its own cheap clone), so
/// a spawned vCPU borrows nothing from its parent and can run on a detached OS thread (the seam for a
/// `'static` worker pool). The shared runtime state — thread `budget`, the `parking` lot, the
/// `registry` of spawned threads — is `Arc`-shared across all vCPUs.
///
/// All the run state is **owned**, so a vCPU is `Send` and self-contained — the basis for moving it
/// A resolved §14 `Module` grant's pieces, as the eval loop carries them: the child module's
/// functions, declared window size, and data segments (`Arc`s — spawning shares, never copies).
type ModArc = (Arc<[Func]>, Option<u8>, Arc<[Data]>);

/// A §14 co-fiber child the parent drives with `resume`: its suspended continuation plus whether it is
/// **awaiting a resume value** — i.e. parked at a `yield` (so the next `resume` delivers its argument
/// as the yield's result) vs. freshly spawned at its entry (the first `resume` just starts it, its
/// argument unused). The child runs *inline* on the parent's thread, never on the executor.
struct Coro {
    vcpu: Box<VCpu>,
    awaiting_resume: bool,
    /// When set, the child is suspended at a **fault-driven yield** awaiting this (confined) page: the
    /// next `resume` supplies it (maps it read-write) before re-running the rewound faulting access.
    faulted_page: Option<u64>,
}

/// between worker threads and (next) parking its continuation on a blocking op.
struct VCpu {
    /// The owned function table; `Frame::func` resolves against it.
    funcs: Arc<[Func]>,
    /// The fiber table (§12). `fibers[0]` is the root; `cont.new` appends. The *running* fiber's
    /// frames live in `frames`; its slot holds `Running`.
    fibers: Vec<Fiber>,
    /// The resume chain: `chain[0]` the root, `chain.last()` the running fiber.
    chain: Vec<usize>,
    /// Index of the running fiber.
    cur: usize,
    /// The running fiber's reified call stack.
    frames: Vec<Frame>,
    /// This vCPU's linear-memory view (shared `Region` + address space; see [`Mem`]).
    mem: Option<Mem>,
    /// The domain's powerbox, **shared** by every vCPU of the run (`Arc<Mutex<Host>>`): a spawned
    /// thread inherits the same capability table + I/O sinks, so a handle granted to the domain works
    /// in any thread and I/O from any thread reaches the same sink (matching the JIT, whose `cap.call`s
    /// all hit the one host ctx). Locked briefly per `cap.call`.
    host: Arc<Mutex<Host>>,
    /// Remaining fuel (metering, §5).
    fuel: u64,
    /// This vCPU's spawned children, by `thread.join` handle (slot) ⇒ child [`TaskId`]; `None` once
    /// joined (a re-join is inert).
    threads: Vec<Option<TaskId>>,
    /// This vCPU's §14 **co-fiber** children (`Instantiator.spawn_coroutine`): suspended continuations
    /// (their own frames/mem/host) driven *inline* by `resume`, by handle (slot). `None` once the
    /// coroutine has run to completion (a later `resume` is inert). Distinct from `threads` — a
    /// coroutine is cooperative (parent and child never run concurrently), not an executor vCPU.
    coroutines: Vec<Option<Coro>>,
    /// §14: this vCPU is a coroutine whose recoverable page faults **suspend to its parent** (fault-
    /// driven yield / lazy paging) instead of trapping. Set for `Instantiator.spawn_coroutine`
    /// children; `false` for every ordinary vCPU (a page fault is detect-and-kill).
    fault_yields: bool,
    /// Call-depth base for the stack-overflow bound.
    depth: u32,
    /// This task's own id (where its outcome is published on completion).
    id: TaskId,
    /// Set when resuming from a park: how to finish the blocked op (see [`Pending`]).
    pending: Option<Pending>,
    /// The executor this vCPU runs under — the real OS-thread [`Scheduler`] or the deterministic
    /// [`DetSched`] (spawn enqueues here; notify wakes here).
    sched: SchedRef,
    /// When set, the `quantum` budget counts **visible (shared-memory / sync) operations** rather than
    /// raw instructions, so the vCPU yields at memory-op boundaries. The exhaustive model checker
    /// ([`explore_all`]) uses `memop = true` + `quantum = 1` to make every shared-state access a
    /// scheduling point; the real pool and the seeded explorer leave it `false`.
    memop: bool,
    /// The object touched by the **visible op this turn ran** (set in `memop` mode at the op's commit
    /// point; `None` if the turn ran no visible op). Read back by the DPOR driver ([`explore_all`]) to
    /// build the schedule trace; unused by the real pool / seeded explorer.
    acc: Option<MemAccess>,
}

impl VCpu {
    /// A fresh vCPU whose root frame is `funcs[entry](args)`. A bad `entry` is caught by the driver's
    /// first block lookup ([`Trap::Malformed`]), so construction is infallible.
    #[allow(clippy::too_many_arguments)]
    fn new(
        funcs: Arc<[Func]>,
        entry: FuncIdx,
        args: &[Value],
        mem: Option<Mem>,
        host: Arc<Mutex<Host>>,
        fuel: u64,
        depth: u32,
        id: TaskId,
        sched: SchedRef,
    ) -> VCpu {
        VCpu {
            funcs,
            fibers: vec![Fiber::Running],
            chain: vec![0],
            cur: 0,
            frames: vec![Frame {
                func: entry,
                block: 0,
                inst: 0,
                vals: args.to_vec(),
            }],
            mem,
            host,
            fuel,
            threads: Vec::new(),
            coroutines: Vec::new(),
            fault_yields: false,
            depth,
            id,
            pending: None,
            sched,
            memop: false,
            acc: None,
        }
    }

    /// A 64-bit fingerprint of this vCPU's **local** execution configuration (its fibers + reified call
    /// stacks: function / block / instruction / SSA values — everything *except* shared memory). The
    /// explorer compares it across one turn: a visible op that returns the vCPU to the same fingerprint
    /// has gone once around a loop with no local progress — a spin (livelock unless shared memory it
    /// reads changes). Collisions would risk a false spin-park, but two configs of the *same* vCPU one
    /// op apart colliding is ~2^-64; the values fully determine the hash (floats by bit pattern).
    fn local_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        fn hash_vals(h: &mut std::collections::hash_map::DefaultHasher, vals: &[Value]) {
            vals.len().hash(h);
            for v in vals {
                match v {
                    Value::I32(x) => (0u8, *x as i64).hash(h),
                    Value::I64(x) => (1u8, *x).hash(h),
                    Value::F32(x) => (2u8, x.to_bits() as u64).hash(h),
                    Value::F64(x) => (3u8, x.to_bits()).hash(h),
                }
            }
        }
        fn hash_frames(h: &mut std::collections::hash_map::DefaultHasher, frames: &[Frame]) {
            frames.len().hash(h);
            for f in frames {
                (f.func, f.block, f.inst).hash(h);
                hash_vals(h, &f.vals);
            }
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.cur.hash(&mut h);
        self.chain.hash(&mut h);
        hash_frames(&mut h, &self.frames);
        // Parked fibers are part of the configuration (a resume could re-enter them).
        self.fibers.len().hash(&mut h);
        for fib in &self.fibers {
            match fib {
                Fiber::Pending { func, sp } => (0u8, *func, *sp).hash(&mut h),
                Fiber::Live(frames) => {
                    1u8.hash(&mut h);
                    hash_frames(&mut h, frames);
                }
                Fiber::Running => 2u8.hash(&mut h),
                Fiber::Done => 3u8.hash(&mut h),
            }
        }
        h.finish()
    }

    /// The current `width`-byte value at confined `key` (no checks; used by the executor for the
    /// futex compare under the scheduler lock). Zero if this vCPU has no memory.
    fn atomic_value(&self, key: u64, width: u32) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.atomic_value(key, width))
    }

    /// Run for up to `quantum` instructions, then finish / park / yield. The real executor passes
    /// `u64::MAX` (run to completion or park); the deterministic explorer passes a small seeded
    /// quantum to interleave vCPUs finely. Folds a trap into `Step::Done(Err)`.
    fn run(&mut self, quantum: u64) -> Step {
        match run_inner(self, quantum) {
            Ok(Inner::Done(v)) => Step::Done(Ok(v)),
            Ok(Inner::Park(b)) => Step::Park(b),
            Ok(Inner::Yield) => Step::Yield,
            // A `Yielder` cap.call / fault-driven yield on a vCPU the *executor* runs has no resumer to
            // yield to (a coroutine child is driven inline by `resume`, never enqueued here) — inert.
            Ok(Inner::CoYield(_)) | Ok(Inner::CoFault(_)) => Step::Done(Err(Trap::FiberFault)),
            Err(t) => Step::Done(Err(t)),
        }
    }
}

/// A **visible** instruction — one whose effect another vCPU can observe or that synchronizes with
/// one: a linear-memory access (atomic or plain) or a thread/futex op. These are the only points at
/// which interleaving order can change a program's outcome, so they are the scheduling decision
/// points the exhaustive model checker preempts on (`memop` granularity). Pure thread-local
/// computation (arithmetic, control flow, calls) is invisible and runs without a yield. `atomic.fence`
/// is omitted: both backends execute seq-cst, so a fence moves no data and adds no observable order.
fn is_visible(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Load { .. }
            | Inst::Store { .. }
            | Inst::AtomicLoad { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicRmw { .. }
            | Inst::AtomicCmpxchg { .. }
            | Inst::ThreadSpawn { .. }
            | Inst::ThreadJoin { .. }
            | Inst::MemoryWait { .. }
            | Inst::MemoryNotify { .. }
    )
}

/// Drive a vCPU until it finishes (`Inner::Done`) or parks on a blocking op (`Inner::Park`). On
/// re-entry it first completes the parked op recorded in `pending`. The owned `funcs` is borrowed
/// locally as `fs` so the loop can mutate the other fields.
fn run_inner(v: &mut VCpu, quantum: u64) -> Result<Inner, Trap> {
    let mut budget = quantum; // instructions left before a forced `Yield` (deterministic explorer)
                              // Resuming from a park: finish the op the scheduler woke us for.
    match v.pending.take() {
        Some(Pending::Join { slot }) => {
            let child = v
                .threads
                .get(slot)
                .and_then(|x| *x)
                .ok_or(Trap::ThreadFault)?;
            v.threads[slot] = None; // a handle is joined once
            let out = v.sched.take_result(child).ok_or(Trap::Malformed)?;
            let vals = out.result?; // a child trap propagates as this vCPU's trap
            let top = v.frames.len() - 1;
            v.frames[top]
                .vals
                .push(vals.first().copied().unwrap_or(Value::I64(0)));
        }
        Some(Pending::Wait(status)) => {
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Value::I32(status));
        }
        Some(Pending::CoResume(value)) => {
            // The parent's `resume` delivered `value` — push it as the child `Yielder` cap.call's
            // result so the coroutine continues past its `yield`.
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Value::I64(value));
        }
        None => {}
    }

    let funcs = Arc::clone(&v.funcs);
    let fs: &[Func] = &funcs;
    let VCpu {
        fibers,
        chain,
        cur,
        frames,
        mem,
        host,
        fuel,
        threads,
        coroutines,
        fault_yields,
        depth,
        pending,
        sched,
        funcs: _,
        id: _,
        memop,
        acc,
    } = v;
    let depth = *depth;
    let memop = *memop;
    let fault_yields = *fault_yields;

    // Drive the running fiber's top frame. A `call` pushes a new top and restarts here; a
    // `return` pops and appends results to the caller (which resumes past the call); a tail
    // call replaces the top in place (O(1) frames). `cont.resume`/`suspend` switch which
    // fiber's stack is in `frames` — see the comments on those arms.
    'frames: loop {
        let top = frames.len() - 1;
        // `block` borrows `fs` (the owned function table), *not* `frames`, so the loop body is free
        // to push/pop/mutate `frames` (and move it on a fiber switch) while holding it.
        let block = fs
            .get(frames[top].func as usize)
            .ok_or(Trap::Malformed)?
            .blocks
            .get(frames[top].block)
            .ok_or(Trap::Malformed)?;

        // Execute the remaining instructions of this block.
        while frames[top].inst < block.insts.len() {
            // Deterministic-explorer preemption: yield at an instruction boundary (state consistent;
            // `inst` not yet advanced) when the quantum is spent. The real pool passes `u64::MAX`.
            // In `memop` mode the budget counts only **visible** (shared-state / sync) ops, so
            // thread-local computation runs to the next memory op before a yield is possible — the
            // partial-order reduction that keeps exhaustive exploration tractable.
            if memop {
                if is_visible(&block.insts[frames[top].inst]) {
                    if budget == 0 {
                        return Ok(Inner::Yield);
                    }
                    // Record the object this visible op touches (for DPOR's race analysis) before it
                    // runs — the confined address is a pure function of the live SSA values here.
                    *acc = Some(access_of(
                        &block.insts[frames[top].inst],
                        &frames[top].vals,
                        mem,
                    ));
                    budget -= 1;
                }
            } else {
                if budget == 0 {
                    return Ok(Inner::Yield);
                }
                budget -= 1;
            }
            let inst = &block.insts[frames[top].inst];
            step(fuel)?;
            frames[top].inst += 1; // advance first, so a call-return resumes past this inst

            match inst {
                // Non-tail calls push a new frame and switch to it; the callee's results
                // are appended to this frame's `vals` when it returns (see `Return`).
                Inst::Call { func, args } => {
                    let argv = collect(&frames[top].vals, args)?;
                    if fs.get(*func as usize).is_none() {
                        return Err(Trap::Malformed);
                    }
                    if depth as usize + live_frames(&fibers[..], &chain[..], frames.len())
                        > MAX_CALL_DEPTH as usize
                    {
                        return Err(Trap::StackOverflow);
                    }
                    frames.push(Frame {
                        func: *func,
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                Inst::CallIndirect { ty, idx, args } => {
                    let callee = table_lookup(fs, as_i32(get(&frames[top].vals, *idx)?)?, ty)?;
                    let argv = collect(&frames[top].vals, args)?;
                    if depth as usize + live_frames(&fibers[..], &chain[..], frames.len())
                        > MAX_CALL_DEPTH as usize
                    {
                        return Err(Trap::StackOverflow);
                    }
                    frames.push(Frame {
                        func: callee,
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                // §14 `Instantiator` (iface 6): serviced here, not in the generic host dispatch, because
                // `instantiate` spawns a child vCPU and `join` parks — both need the executor. The
                // handle still gates authority (resolve as Instantiator → its carve range `[base,
                // base+size)`); a forged/wrong-type handle is an inert `CapFault`.
                // §14 co-fiber `yield` (iface 7): suspend this (coroutine) child, handing `value` to
                // the instantiator-parent's `resume`. Serviced here — it must yield the running
                // continuation, which the generic dispatch can't. The cap.call's result (the resumed
                // value) is delivered on the next `resume` via `Pending::CoResume`; the inst pointer is
                // already advanced, so we return `CoYield` without pushing a result.
                Inst::CapCall {
                    type_id: iface::YIELDER,
                    op,
                    handle,
                    args,
                    ..
                } => {
                    if *op != 0 {
                        return Err(Trap::CapFault);
                    }
                    let h = as_i32(get(&frames[top].vals, *handle)?)?;
                    {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_yielder(h)?; // authority: a forged/wrong handle is inert
                    }
                    let value = as_i64(get(
                        &frames[top].vals,
                        *args.first().ok_or(Trap::Malformed)?,
                    )?)?;
                    return Ok(Inner::CoYield(value));
                }
                // §13/§14 cross-domain **`grant`** (SharedRegion op 4): install this region — the
                // *same* shared backing — into a suspended coroutine child's powerbox, returning the
                // handle the **child** will use. Serviced here (the generic dispatch can't reach the
                // coroutine table). The parent delivers the returned handle to the child by existing
                // means (typically the next `resume`'s value); the child `map`s the region into its
                // own window — the zero-copy cross-domain data plane (§13). Executor (`instantiate`)
                // children and the JIT parent are follow-ups; a forged region handle or an
                // unknown/finished child is an inert `CapFault`.
                Inst::CapCall {
                    type_id: iface::SHARED_REGION,
                    op: 4,
                    handle,
                    args,
                    ..
                } => {
                    let h = as_i32(get(&frames[top].vals, *handle)?)?;
                    let backing = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_region(h)?
                    };
                    let ch = as_i32(get(
                        &frames[top].vals,
                        *args.first().ok_or(Trap::Malformed)?,
                    )?)?;
                    let coro = coroutines
                        .get_mut(ch as usize)
                        .and_then(|c| c.as_mut())
                        .ok_or(Trap::CapFault)?;
                    // Install the region into the child's powerbox. Guest-minting into the *child*
                    // table, so a full table yields -EMFILE rather than panicking (§3c / audit #1).
                    let child_handle = {
                        let mut chh = coro.vcpu.host.lock().unwrap_or_else(|e| e.into_inner());
                        chh.try_grant_shared_region_backed(backing)
                    };
                    frames[top]
                        .vals
                        .push(Value::I64(child_handle.map_or(EMFILE, |h| h as i64)));
                }
                Inst::CapCall {
                    type_id: iface::INSTANTIATOR,
                    op,
                    handle,
                    args,
                    ..
                } => {
                    let h = as_i32(get(&frames[top].vals, *handle)?)?;
                    let (ibase, isize) = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_instantiator(h)?
                    };
                    // §14 **separate-module children** (ops 5/6/7 = `instantiate_module` /
                    // `spawn_coroutine_module` / `spawn_demand_coroutine_module`): exactly ops 0/2/4,
                    // except the first arg is a host-granted `Module` handle (iface 8) and the child
                    // domain runs *that* verified module — the "plugin-in-plugin" story (a guest can
                    // only instantiate modules it was given). Resolve the grant here, shift the
                    // remaining args by one, and fold into the shared op logic below; `join`/`resume`
                    // (ops 1/3) serve both kinds unchanged. A forged module handle is a `CapFault`.
                    let (op, child_mod, askip): (u32, Option<ModArc>, usize) = match *op {
                        mop @ 5..=7 => {
                            // The module handle crosses as an i64 arg (the slot ABI); low 32 bits.
                            let mh = as_i64(get(
                                &frames[top].vals,
                                *args.first().ok_or(Trap::Malformed)?,
                            )?)? as i32;
                            let g = {
                                let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                let g = hg.resolve_module(mh)?;
                                (g.funcs.clone(), g.memory_log2, g.data.clone())
                            };
                            (
                                match mop {
                                    5 => 0, // instantiate_module → instantiate
                                    6 => 2, // spawn_coroutine_module → spawn_coroutine
                                    _ => 4, // spawn_demand_coroutine_module → spawn_demand_coroutine
                                },
                                Some(g),
                                1,
                            )
                        }
                        o => (o, None, 0),
                    };
                    // The function table the child's `entry` indexes — its own module's, or ours.
                    let cfs: &[Func] = child_mod.as_ref().map_or(fs, |(f, _, _)| f);
                    match op {
                        // instantiate(entry, off, size_log2, fuel) -> child handle (or -EINVAL). The
                        // §14 data plane is shared memory: the parent seeds the child's sub-window
                        // directly (it sees the superset) before/after, so there is no scalar arg.
                        0 => {
                            let argn = |i: usize| -> Result<i64, Trap> {
                                as_i64(get(
                                    &frames[top].vals,
                                    *args.get(i + askip).ok_or(Trap::Malformed)?,
                                )?)
                            };
                            let entry = argn(0)? as u64;
                            let off = argn(1)? as u64;
                            let size_log2 = argn(2)?;
                            let quota = argn(3)?;
                            // The child entry returns one `i64` and takes either one `i64` (its
                            // `Instantiator`) or two (its `Instantiator`, its `AddressSpace`) — its
                            // starter capabilities, both over its own window. A missing/mistyped entry
                            // is rejected, not run.
                            let want_as = cfs
                                .get(entry as usize)
                                .is_some_and(|f| f.params == [ValType::I64, ValType::I64]);
                            let ok_entry = cfs.get(entry as usize).is_some_and(|f| {
                                f.results == [ValType::I64]
                                    && (f.params == [ValType::I64]
                                        || f.params == [ValType::I64, ValType::I64])
                            });
                            // The carve must be a power-of-two-aligned sub-window within `[0, isize)`
                            // — a child can only get what the holder sub-allocates (§14/D19). A
                            // separate-module child's carve must **equal its declared memory** (§14
                            // transparency: the plugin runs exactly as it would standalone — same
                            // window size, same wrap behaviour; a module with no memory can't nest).
                            let child_size = if (0..64).contains(&size_log2) {
                                1u64 << size_log2
                            } else {
                                0
                            };
                            let mod_ok = child_mod
                                .as_ref()
                                .is_none_or(|(_, ml, _)| *ml == Some(size_log2 as u8));
                            let fits = child_size != 0
                                && child_size <= isize
                                && off & (child_size - 1) == 0
                                && off.checked_add(child_size).is_some_and(|e| e <= isize);
                            if !ok_entry || !fits || !mod_ok {
                                frames[top].vals.push(Value::I32(EINVAL as i32));
                            } else {
                                // `ibase`/`off` are holder-relative; the backing-absolute base
                                // adds the holder's own window base (0 for a top-level holder), so
                                // nesting composes at any depth.
                                let abs_base =
                                    mem.as_ref().map_or(0, |m| m.window.base()) + ibase + off;
                                let child_mem = mem
                                    .as_ref()
                                    .map(|m| m.nested_view(abs_base, size_log2 as u8));
                                // A separate-module child's **data segments** materialize into the
                                // carve at spawn (exactly as if the child wrote them; the verifier
                                // bounded them to its declared window == the carve). RO protection of
                                // `readonly` segments is skipped for nested children (documented —
                                // intra-domain self-corruption is a §1 non-goal).
                                if let (Some((_, _, data)), Some(m)) = (&child_mod, mem.as_ref()) {
                                    for d in data.iter() {
                                        if d.offset.saturating_add(d.bytes.len() as u64)
                                            <= child_size
                                        {
                                            for (k, &b) in d.bytes.iter().enumerate() {
                                                m.set_byte(abs_base + d.offset + k as u64, b);
                                            }
                                        }
                                    }
                                }
                                // Attenuated powerbox: the child gets, over its *own* window (a strict
                                // subset of the parent's authority), an `Instantiator` (so it can
                                // itself nest — confinement composes to any depth) and an
                                // `AddressSpace` (so it can manage its own pages). These are its entry
                                // arguments. (Pass-through of the parent's *other* handles is a
                                // follow-up.)
                                let mut ch = Host::new();
                                let cinst = ch.grant_instantiator(0, child_size);
                                let cas = ch.grant_address_space(0, child_size);
                                let child_host = Arc::new(Mutex::new(ch));
                                let child_args = if want_as {
                                    vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                                } else {
                                    vec![Value::I64(cinst as i64)]
                                };
                                // Quota: the child's fuel, sub-allocated from (and capped by) ours.
                                let child_fuel = if quota <= 0 {
                                    *fuel
                                } else {
                                    (quota as u64).min(*fuel)
                                };
                                let cfuncs = child_mod
                                    .as_ref()
                                    .map_or_else(|| Arc::clone(&funcs), |(f, _, _)| Arc::clone(f));
                                let csched = sched.clone();
                                let made = sched.spawn(move |id| {
                                    let mut child = VCpu::new(
                                        cfuncs,
                                        entry as u32,
                                        &child_args, // [Instantiator] or [Instantiator, AddressSpace]
                                        child_mem,
                                        child_host,
                                        child_fuel,
                                        depth + 1,
                                        id,
                                        csched,
                                    );
                                    child.memop = memop;
                                    Box::new(child)
                                });
                                match made {
                                    Some(child_id) => {
                                        threads.push(Some(child_id));
                                        frames[top]
                                            .vals
                                            .push(Value::I32((threads.len() - 1) as i32));
                                    }
                                    None => return Err(Trap::ThreadFault),
                                }
                            }
                        }
                        // join(child) -> result: park only this fiber until the child finishes (its
                        // result/trap is delivered on resume via `Pending::Join`); siblings run on.
                        1 => {
                            let ch = as_i32(get(
                                &frames[top].vals,
                                *args.first().ok_or(Trap::Malformed)?,
                            )?)?;
                            let slot = resolve_thread(threads, ch)?;
                            let child = threads[slot].ok_or(Trap::ThreadFault)?;
                            *pending = Some(Pending::Join { slot });
                            return Ok(Inner::Park(Blocked::Join { child }));
                        }
                        // spawn_coroutine (op 2) / spawn_demand_coroutine (op 4) (entry, off,
                        // size_log2, fuel) -> child handle (or -EINVAL). Like instantiate, but the child
                        // is a **suspended coroutine** (its own confined window + a `Yielder` handle
                        // back to us, its single entry arg), driven cooperatively by `resume` — not run
                        // on the executor. op 4 additionally **demand-pages** the child's window (every
                        // page starts unmapped), so the child faults on first access and we supply the
                        // page — the §14 parent-virtualized-fault / userfaultfd-style lazy-paging model.
                        2 | 4 => {
                            let demand = op == 4;
                            let argn = |i: usize| -> Result<i64, Trap> {
                                as_i64(get(
                                    &frames[top].vals,
                                    *args.get(i + askip).ok_or(Trap::Malformed)?,
                                )?)
                            };
                            let entry = argn(0)? as u64;
                            let off = argn(1)? as u64;
                            let size_log2 = argn(2)?;
                            let _quota = argn(3)?; // (per-coroutine fuel metering is a follow-up)
                                                   // A coroutine child entry is a fixed `(i64 yielder) -> (i64)`.
                            let ok_entry = cfs.get(entry as usize).is_some_and(|f| {
                                f.params == [ValType::I64] && f.results == [ValType::I64]
                            });
                            let child_size = if (0..64).contains(&size_log2) {
                                1u64 << size_log2
                            } else {
                                0
                            };
                            // A separate-module child's carve must equal its declared memory (§14
                            // transparency), exactly as for `instantiate_module`.
                            let mod_ok = child_mod
                                .as_ref()
                                .is_none_or(|(_, ml, _)| *ml == Some(size_log2 as u8));
                            let fits = child_size != 0
                                && child_size <= isize
                                && off & (child_size - 1) == 0
                                && off.checked_add(child_size).is_some_and(|e| e <= isize);
                            if !ok_entry || !fits || !mod_ok {
                                frames[top].vals.push(Value::I32(EINVAL as i32));
                            } else {
                                // `ibase`/`off` are holder-relative; the backing-absolute base
                                // adds the holder's own window base (0 for a top-level holder), so
                                // nesting composes at any depth.
                                let abs_base =
                                    mem.as_ref().map_or(0, |m| m.window.base()) + ibase + off;
                                let child_mem = mem.as_ref().map(|m| {
                                    let cm = m.nested_view(abs_base, size_log2 as u8);
                                    if demand {
                                        cm.demand_page(); // every page starts unmapped (lazy paging)
                                    }
                                    cm
                                });
                                // A separate-module child's data segments materialize into the carve
                                // at spawn (see `instantiate`). For a **demand** coroutine they land
                                // in the parent's backing while the child's pages start unmapped — so
                                // a plugin's data segments are *supplied lazily*, page by page, as it
                                // first touches them (the §14 parent-as-pager model, for free).
                                if let (Some((_, _, data)), Some(m)) = (&child_mod, mem.as_ref()) {
                                    for d in data.iter() {
                                        if d.offset.saturating_add(d.bytes.len() as u64)
                                            <= child_size
                                        {
                                            for (k, &b) in d.bytes.iter().enumerate() {
                                                m.set_byte(abs_base + d.offset + k as u64, b);
                                            }
                                        }
                                    }
                                }
                                let mut ch = Host::new();
                                let cy = ch.grant_yielder(); // the child's handle to suspend back to us
                                let child_host = Arc::new(Mutex::new(ch));
                                let cfuncs = child_mod
                                    .as_ref()
                                    .map_or_else(|| Arc::clone(&funcs), |(f, _, _)| Arc::clone(f));
                                let mut child = VCpu::new(
                                    cfuncs,
                                    entry as u32,
                                    &[Value::I64(cy as i64)],
                                    child_mem,
                                    child_host,
                                    *fuel,
                                    depth + 1,
                                    0, // unused: a coroutine is driven inline, never via the executor
                                    sched.clone(),
                                );
                                child.fault_yields = true; // its page faults suspend to us, not trap
                                coroutines.push(Some(Coro {
                                    vcpu: Box::new(child),
                                    awaiting_resume: false,
                                    faulted_page: None,
                                }));
                                frames[top]
                                    .vals
                                    .push(Value::I32((coroutines.len() - 1) as i32));
                            }
                        }
                        // resume(child, value) -> (status: i32, value: i64): drive the coroutine
                        // **inline** until it `yield`s (SUSPENDED), faults on an unmapped page (FAULTED,
                        // value = fault address), or returns (RETURNED). The first resume starts it (its
                        // `value` arg unused); a resume after an explicit yield delivers `value` as the
                        // yield's result; a resume after a fault first **supplies** the faulted page
                        // (the parent has placed its bytes in the shared window) and re-runs the access.
                        // A child trap propagates to us.
                        3 => {
                            let ch = as_i32(get(
                                &frames[top].vals,
                                *args.first().ok_or(Trap::Malformed)?,
                            )?)?;
                            let value = as_i64(get(
                                &frames[top].vals,
                                *args.get(1).ok_or(Trap::Malformed)?,
                            )?)?;
                            let slot = ch as usize;
                            let mut coro = match coroutines.get_mut(slot).and_then(|c| c.take()) {
                                Some(c) => c,
                                None => return Err(Trap::CapFault), // forged or already-finished
                            };
                            if let Some(addr) = coro.faulted_page.take() {
                                // Supply the faulted page (map it RW, keeping the parent's bytes), then
                                // re-run the rewound access.
                                if let Some(m) = &coro.vcpu.mem {
                                    m.supply_page(addr);
                                }
                            } else if coro.awaiting_resume {
                                coro.vcpu.pending = Some(Pending::CoResume(value));
                            }
                            match run_inner(&mut coro.vcpu, u64::MAX) {
                                Ok(Inner::CoYield(yv)) => {
                                    coro.awaiting_resume = true;
                                    coroutines[slot] = Some(coro);
                                    frames[top].vals.push(Value::I32(FIBER_SUSPENDED));
                                    frames[top].vals.push(Value::I64(yv));
                                }
                                Ok(Inner::CoFault(addr)) => {
                                    coro.faulted_page = Some(addr);
                                    coroutines[slot] = Some(coro);
                                    frames[top].vals.push(Value::I32(CORO_FAULTED));
                                    frames[top].vals.push(Value::I64(addr as i64));
                                }
                                Ok(Inner::Done(result)) => {
                                    // Finished — `coroutines[slot]` stays `None` (a later resume inert).
                                    frames[top].vals.push(Value::I32(FIBER_RETURNED));
                                    frames[top]
                                        .vals
                                        .push(result.first().copied().unwrap_or(Value::I64(0)));
                                }
                                // A coroutine that parks used a blocking concurrency op (it has no
                                // executor driving it) — unsupported; surface as a fault.
                                Ok(Inner::Park(_)) | Ok(Inner::Yield) => {
                                    return Err(Trap::FiberFault)
                                }
                                Err(t) => return Err(t), // a child trap propagates to the parent
                            }
                        }
                        _ => return Err(Trap::CapFault),
                    }
                }
                Inst::CapCall {
                    type_id,
                    op,
                    sig,
                    handle,
                    args,
                } => {
                    // Capability call (§3c): resolve the handle in the host-owned table
                    // (mask + type_id/generation check) and dispatch to the mock host.
                    // Args/results cross as i64 slots (the shared host-dispatch ABI).
                    // Synchronous in the reference (the async/submit-complete ABI is §12).
                    let h = as_i32(get(&frames[top].vals, *handle)?)?;
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(val_to_slot(get(&frames[top].vals, *a)?));
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    // Lock the shared powerbox for the duration of this one cap.call (brief; no nested
                    // host locking). Threads of a domain serialize their capability calls here.
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results = hg.cap_dispatch_slots(*type_id, *op, h, &argv, gm)?;
                    for (s, ty) in results.iter().zip(&sig.results) {
                        frames[top].vals.push(slot_to_val(*ty, *s));
                    }
                }
                // §12 fiber create: record a `Pending` fiber, yield its handle. No switch.
                Inst::ContNew { func, sp } => {
                    let funcref = as_i32(get(&frames[top].vals, *func)?)?;
                    let stack_base = as_i64(get(&frames[top].vals, *sp)?)?;
                    if fibers.len() >= MAX_FIBERS {
                        return Err(Trap::FiberFault);
                    }
                    let handle = fibers.len() as i32;
                    fibers.push(Fiber::Pending {
                        func: funcref,
                        sp: stack_base,
                    });
                    frames[top].vals.push(Value::I32(handle));
                }
                // §12 fiber resume: switch into fiber `k`, delivering `arg`. The two results
                // `(status, value)` are appended to *this* frame later, when `k` suspends or
                // returns control here (see `Suspend` and `Return`).
                Inst::ContResume { k, arg } => {
                    let kh = as_i32(get(&frames[top].vals, *k)?)?;
                    let av = as_i64(get(&frames[top].vals, *arg)?)?;
                    let target = resolve_fiber(&fibers[..], &chain[..], kh)?;
                    // Materialize the target's frames: start a `Pending` fiber, or continue a
                    // parked one (delivering `arg` as the result of its `suspend`).
                    let new_frames = match std::mem::replace(&mut fibers[target], Fiber::Running) {
                        Fiber::Pending { func: funcref, sp } => {
                            let callee = table_lookup(fs, funcref, &fiber_sig())?;
                            // First entry: call `func(sp, arg)` on the fiber's data stack.
                            vec![Frame {
                                func: callee,
                                block: 0,
                                inst: 0,
                                vals: vec![Value::I64(sp), Value::I64(av)],
                            }]
                        }
                        Fiber::Live(mut f) => {
                            f.last_mut()
                                .ok_or(Trap::Malformed)?
                                .vals
                                .push(Value::I64(av));
                            f
                        }
                        // `resolve_fiber` already rejected Running/Done.
                        _ => return Err(Trap::FiberFault),
                    };
                    // Park the resumer and switch to the target.
                    fibers[*cur] = Fiber::Live(std::mem::take(frames));
                    chain.push(target);
                    *cur = target;
                    *frames = new_frames;
                    continue 'frames;
                }
                // §12 fiber suspend: hand `value` back to the resumer with status SUSPENDED;
                // park this fiber (its `suspend` result pends until the next resume).
                Inst::Suspend { value } => {
                    if chain.len() == 1 {
                        return Err(Trap::FiberFault); // no resumer (the root cannot suspend)
                    }
                    let v = as_i64(get(&frames[top].vals, *value)?)?;
                    fibers[*cur] = Fiber::Live(std::mem::take(frames));
                    chain.pop();
                    *cur = *chain.last().expect("chain keeps the root");
                    *frames = take_running(&mut fibers[..], *cur)?;
                    let rtop = frames.len() - 1;
                    frames[rtop].vals.push(Value::I32(FIBER_SUSPENDED));
                    frames[rtop].vals.push(Value::I64(v));
                    continue 'frames;
                }
                // §12 thread spawn: enqueue a new vCPU (green thread) running `funcs[func](arg)` over
                // the *shared* memory (the `Arc<Region>` bytes + §13 `Arc` regions; the child snapshots
                // the page-protection map). The executor runs it on a pooled worker. The child **shares
                // the domain's powerbox** (the same `Arc<Mutex<Host>>`), so a handle granted to the
                // domain works in the child and its I/O reaches the same sink; it gets its own fuel.
                // Yields an i32 thread handle (the table slot).
                Inst::ThreadSpawn { func, sp, arg } => {
                    if fs.get(*func as usize).is_none() {
                        return Err(Trap::Malformed);
                    }
                    let entry = *func;
                    let spv = as_i64(get(&frames[top].vals, *sp)?)?; // the thread's data-stack base
                    let av = as_i64(get(&frames[top].vals, *arg)?)?;
                    let child_mem = mem.as_ref().map(|m| m.fork_for_thread());
                    let child_host = Arc::clone(host); // inherit the domain powerbox
                    let child_fuel = *fuel; // the child's own metering budget (a copy)
                    let cfuncs = Arc::clone(&funcs);
                    let csched = sched.clone();
                    let made = sched.spawn(move |id| {
                        let mut child = VCpu::new(
                            cfuncs,
                            entry,
                            &[Value::I64(spv), Value::I64(av)], // (sp, arg) — the fiber-style entry
                            child_mem,
                            child_host,
                            child_fuel,
                            0,
                            id,
                            csched,
                        );
                        child.memop = memop; // inherit the explorer's memory-op granularity
                        Box::new(child)
                    });
                    match made {
                        Some(child_id) => {
                            threads.push(Some(child_id));
                            frames[top]
                                .vals
                                .push(Value::I32((threads.len() - 1) as i32));
                        }
                        None => return Err(Trap::ThreadFault), // live cap (a thread-bomb)
                    }
                }
                // §12 thread join: park until vCPU `handle` finishes, then (on resume) take its i64
                // result. A forged / out-of-range / already-joined handle is inert (masked + checked
                // like a fiber handle); a trap in the joined vCPU propagates here (on resume).
                Inst::ThreadJoin { handle } => {
                    let h = as_i32(get(&frames[top].vals, *handle)?)?;
                    let slot = resolve_thread(threads, h)?;
                    let child = threads[slot].ok_or(Trap::ThreadFault)?;
                    *pending = Some(Pending::Join { slot });
                    return Ok(Inner::Park(Blocked::Join { child }));
                }
                // §12 futex wait: validate the address (confine/align/prot — traps surface here), then
                // park; the executor re-checks the value under its lock (atomic vs. `notify`) and either
                // resumes immediately (value changed → status 1) or blocks until notified / timed out.
                Inst::MemoryWait {
                    ty,
                    addr,
                    expected,
                    timeout,
                } => {
                    let width = atomic_width(*ty);
                    let a = as_i64(get(&frames[top].vals, *addr)?)? as u64;
                    let exp = store_bits(get(&frames[top].vals, *expected)?) & width_mask(width);
                    let to_ns = as_i64(get(&frames[top].vals, *timeout)?)?;
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base = m.prepare_wait(a, *ty)?;
                    let wait = if to_ns < 0 {
                        MAX_WAIT
                    } else {
                        Duration::from_nanos(to_ns as u64).min(MAX_WAIT)
                    };
                    return Ok(Inner::Park(Blocked::Wait {
                        key: base,
                        expected: exp,
                        width,
                        timeout_ns: wait.as_nanos() as u64,
                    }));
                }
                // §12 futex notify: wake up to `count` vCPUs parked on the confined address.
                Inst::MemoryNotify { addr, count } => {
                    let a = as_i64(get(&frames[top].vals, *addr)?)? as u64;
                    let n = as_i32(get(&frames[top].vals, *count)?)?.max(0) as u32;
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base = m.confine_for_notify(a)?;
                    frames[top]
                        .vals
                        .push(Value::I32(sched.notify(base, n) as i32));
                }
                // Everything else: one value, or none for `Store`/`AtomicStore`.
                other => {
                    match eval_inst(other, &frames[top].vals, mem) {
                        Ok(Some(v)) => frames[top].vals.push(v),
                        Ok(None) => {}
                        // §14 fault-driven yield: a coroutine child's access to an unmapped page in its
                        // window suspends to the parent (which supplies the page) instead of trapping.
                        // `take_fault` is `Some` only for a *recoverable* in-window page fault (not an
                        // out-of-window fault, which traps). Rewind to re-execute the access on resume.
                        Err(Trap::MemoryFault) if fault_yields => {
                            match mem.as_ref().and_then(|m| m.take_fault()) {
                                Some(addr) => {
                                    frames[top].inst -= 1;
                                    return Ok(Inner::CoFault(addr));
                                }
                                None => return Err(Trap::MemoryFault),
                            }
                        }
                        Err(t) => return Err(t),
                    }
                }
            }
        }

        step(fuel)?;
        match &block.term {
            Terminator::Br { target, args } => {
                frames[top].vals = collect(&frames[top].vals, args)?;
                frames[top].block = *target as usize;
                frames[top].inst = 0;
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (target, edge_args) = if as_i32(get(&frames[top].vals, *cond)?)? != 0 {
                    (*then_blk, then_args)
                } else {
                    (*else_blk, else_args)
                };
                frames[top].vals = collect(&frames[top].vals, edge_args)?;
                frames[top].block = target as usize;
                frames[top].inst = 0;
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let i = as_i32(get(&frames[top].vals, *idx)?)? as u32 as usize;
                let (target, edge_args) = targets.get(i).unwrap_or(default);
                frames[top].vals = collect(&frames[top].vals, edge_args)?;
                frames[top].block = *target as usize;
                frames[top].inst = 0;
            }
            Terminator::Return(out) => {
                let results = collect(&frames[top].vals, out)?;
                frames.pop();
                if let Some(caller) = frames.last_mut() {
                    // Caller in the same fiber resumes past its `call` (`inst` already advanced).
                    caller.vals.extend(results);
                } else if *cur == 0 {
                    return Ok(Inner::Done(results)); // the root returned: this vCPU is done
                } else {
                    // A fiber's function returned: hand its single `i64` back to the resumer
                    // with status RETURNED; the fiber is now `Done` (resuming again traps).
                    fibers[*cur] = Fiber::Done;
                    chain.pop();
                    *cur = *chain.last().expect("chain keeps the root");
                    *frames = take_running(&mut fibers[..], *cur)?;
                    let rtop = frames.len() - 1;
                    frames[rtop].vals.push(Value::I32(FIBER_RETURNED));
                    frames[rtop]
                        .vals
                        .push(results.into_iter().next().unwrap_or(Value::I64(0)));
                }
            }
            Terminator::Unreachable => return Err(Trap::Unreachable),
            // Tail calls replace the top frame in place — no depth growth.
            Terminator::ReturnCall { func, args } => {
                let argv = collect(&frames[top].vals, args)?;
                if fs.get(*func as usize).is_none() {
                    return Err(Trap::Malformed);
                }
                frames[top] = Frame {
                    func: *func,
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
            Terminator::ReturnCallIndirect { ty, idx, args } => {
                let callee = table_lookup(fs, as_i32(get(&frames[top].vals, *idx)?)?, ty)?;
                let argv = collect(&frames[top].vals, args)?;
                frames[top] = Frame {
                    func: callee,
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
        }
    }
}

fn eval_inst(inst: &Inst, vals: &[Value], mem: &mut Option<Mem>) -> Result<Option<Value>, Trap> {
    // `Store` is the only instruction that produces no value.
    if let Inst::Store {
        op,
        addr,
        value,
        offset,
        ..
    } = inst
    {
        let m = mem.as_mut().ok_or(Trap::Malformed)?;
        let a = as_i64(get(vals, *addr)?)? as u64;
        let v = get(vals, *value)?;
        m.store(a, *offset, *op, v)?;
        return Ok(None);
    }
    // §12 atomic store — the other no-result memory op.
    if let Inst::AtomicStore {
        ty,
        addr,
        value,
        offset,
        ..
    } = inst
    {
        let m = mem.as_mut().ok_or(Trap::Malformed)?;
        let a = as_i64(get(vals, *addr)?)? as u64;
        let v = get(vals, *value)?;
        m.atomic_store(a, *offset, *ty, v)?;
        return Ok(None);
    }
    let v = match inst {
        Inst::ConstI32(c) => Value::I32(*c),
        Inst::ConstI64(c) => Value::I64(*c),
        Inst::IntBin { ty, op, a, b } => match ty {
            IntTy::I32 => Value::I32(bin32(
                *op,
                as_i32(get(vals, *a)?)?,
                as_i32(get(vals, *b)?)?,
            )?),
            IntTy::I64 => Value::I64(bin64(
                *op,
                as_i64(get(vals, *a)?)?,
                as_i64(get(vals, *b)?)?,
            )?),
        },
        Inst::IntCmp { ty, op, a, b } => {
            let r = match ty {
                IntTy::I32 => cmp32(*op, as_i32(get(vals, *a)?)?, as_i32(get(vals, *b)?)?),
                IntTy::I64 => cmp64(*op, as_i64(get(vals, *a)?)?, as_i64(get(vals, *b)?)?),
            };
            Value::I32(r as i32)
        }
        Inst::IntUn { ty, op, a } => match ty {
            IntTy::I32 => Value::I32(intun32(*op, as_i32(get(vals, *a)?)?)),
            IntTy::I64 => Value::I64(intun64(*op, as_i64(get(vals, *a)?)?)),
        },
        Inst::Eqz { ty, a } => {
            let r = match ty {
                IntTy::I32 => as_i32(get(vals, *a)?)? == 0,
                IntTy::I64 => as_i64(get(vals, *a)?)? == 0,
            };
            Value::I32(r as i32)
        }
        Inst::Convert { op, a } => match op {
            ConvOp::ExtendI32S => Value::I64(as_i32(get(vals, *a)?)? as i64),
            ConvOp::ExtendI32U => Value::I64(as_i32(get(vals, *a)?)? as u32 as i64),
            ConvOp::WrapI64 => Value::I32(as_i64(get(vals, *a)?)? as i32),
        },
        Inst::Select { cond, a, b } => {
            if as_i32(get(vals, *cond)?)? != 0 {
                get(vals, *a)?
            } else {
                get(vals, *b)?
            }
        }
        Inst::ConstF32(bits) => Value::F32(f32::from_bits(*bits)),
        Inst::ConstF64(bits) => Value::F64(f64::from_bits(*bits)),
        Inst::FBin { ty, op, a, b } => match ty {
            FloatTy::F32 => Value::F32(fbin32(
                *op,
                as_f32(get(vals, *a)?)?,
                as_f32(get(vals, *b)?)?,
            )),
            FloatTy::F64 => Value::F64(fbin64(
                *op,
                as_f64(get(vals, *a)?)?,
                as_f64(get(vals, *b)?)?,
            )),
        },
        Inst::FUn { ty, op, a } => match ty {
            FloatTy::F32 => Value::F32(fun32(*op, as_f32(get(vals, *a)?)?)),
            FloatTy::F64 => Value::F64(fun64(*op, as_f64(get(vals, *a)?)?)),
        },
        Inst::FCmp { ty, op, a, b } => {
            let r = match ty {
                FloatTy::F32 => fcmp32(*op, as_f32(get(vals, *a)?)?, as_f32(get(vals, *b)?)?),
                FloatTy::F64 => fcmp64(*op, as_f64(get(vals, *a)?)?, as_f64(get(vals, *b)?)?),
            };
            Value::I32(r as i32)
        }
        Inst::FToISat { op, a } => fto_i(*op, get(vals, *a)?)?,
        Inst::FToITrap { op, a } => trunc_trap(*op, get(vals, *a)?)?,
        Inst::IToFConv { op, a } => i_to_f(*op, get(vals, *a)?)?,
        Inst::PtrAdd { a, b } => {
            Value::I64(as_i64(get(vals, *a)?)?.wrapping_add(as_i64(get(vals, *b)?)?))
        }
        // `ptr.from_int`/`ptr.to_int` are a no-op off-CHERI: pass the i64 through.
        Inst::PtrCast { a, .. } => Value::I64(as_i64(get(vals, *a)?)?),
        Inst::Cast { op, a } => cast(*op, get(vals, *a)?)?,
        // A funcref is just the function index as plain i32 data (§3c).
        Inst::RefFunc { func } => Value::I32(*func as i32),
        Inst::Load {
            op, addr, offset, ..
        } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            m.load(a, *offset, *op)?
        }
        // The `order` is carried but execution is seq-cst (a sound strengthening; see `svm_ir::Ordering`).
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            m.atomic_load(a, *offset, *ty)?
        }
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            let v = get(vals, *value)?;
            m.atomic_rmw(a, *offset, *ty, *op, v)?
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            let exp = get(vals, *expected)?;
            let rep = get(vals, *replacement)?;
            m.atomic_cmpxchg(a, *offset, *ty, exp, rep)?
        }
        // §12 standalone fence — issue the real hardware fence (a `Relaxed` fence is a no-op; `std`
        // would panic on it). Both backends are otherwise seq-cst, so this is the one place ordering
        // is observable.
        Inst::AtomicFence { order } => {
            use std::sync::atomic::{fence, Ordering as O};
            match order {
                svm_ir::Ordering::Relaxed => {}
                svm_ir::Ordering::Acquire => fence(O::Acquire),
                svm_ir::Ordering::Release => fence(O::Release),
                svm_ir::Ordering::AcqRel => fence(O::AcqRel),
                svm_ir::Ordering::SeqCst => fence(O::SeqCst),
            }
            return Ok(None);
        }
        // Handled before/around the match (or in `run_func` for the §12 fiber ops, which
        // switch stacks); listed for exhaustiveness (no panic).
        Inst::Store { .. }
        | Inst::AtomicStore { .. }
        | Inst::Call { .. }
        | Inst::CallIndirect { .. }
        | Inst::CapCall { .. }
        | Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. } => return Ok(None),
    };
    Ok(Some(v))
}

/// Resolve a `call_indirect`: mask the index into the power-of-two-padded function
/// table, then check the selected entry's signature against `ty` (the §3c table
/// type-id check). Masking — not branching — keeps the table load Spectre-v1 safe.
fn table_lookup(funcs: &[Func], idx: i32, ty: &FuncType) -> Result<FuncIdx, Trap> {
    let mask = funcs.len().next_power_of_two() - 1;
    let slot = (idx as u32 as usize) & mask;
    match funcs.get(slot) {
        Some(c) if c.params == ty.params && c.results == ty.results => Ok(slot as FuncIdx),
        _ => Err(Trap::IndirectCallType),
    }
}

fn fbin32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin32(a, b),
        FBinOp::Max => fmax32(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fbin64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin64(a, b),
        FBinOp::Max => fmax64(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fun32(op: FUnOp, a: f32) -> f32 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fun64(op: FUnOp, a: f64) -> f64 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fcmp32(op: FCmpOp, a: f32, b: f32) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

fn fcmp64(op: FCmpOp, a: f64, b: f64) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

// wasm min/max: NaN propagates; for ±0, min prefers -0 and max prefers +0.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

// Float→int casts are saturating with NaN→0 (Rust `as` matches wasm `trunc_sat`).
fn fto_i(op: FToI, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        FToI::F32I32S => Value::I32(as_f32(v)? as i32),
        FToI::F32I32U => Value::I32(as_f32(v)? as u32 as i32),
        FToI::F32I64S => Value::I64(as_f32(v)? as i64),
        FToI::F32I64U => Value::I64(as_f32(v)? as u64 as i64),
        FToI::F64I32S => Value::I32(as_f64(v)? as i32),
        FToI::F64I32U => Value::I32(as_f64(v)? as u32 as i32),
        FToI::F64I64S => Value::I64(as_f64(v)? as i64),
        FToI::F64I64U => Value::I64(as_f64(v)? as u64 as i64),
    })
}

/// Trapping float→int conversion (`trunc`, vs the saturating `trunc_sat`): NaN and
/// out-of-range inputs trap. Work in `f64` (promoting `f32` is exact), and trap
/// unless the truncation toward zero fits the target — `f > MIN-1 && f < MAX+1`
/// (using the exact float boundary constants; the `i64` signed lower bound is
/// closed because `-2^63 - 1` is not representable and rounds to `-2^63`).
fn trunc_trap(op: FToI, v: Value) -> Result<Value, Trap> {
    let (from, to, signed) = op.parts();
    let f: f64 = match from {
        FloatTy::F32 => as_f32(v)? as f64,
        FloatTy::F64 => as_f64(v)?,
    };
    if f.is_nan() {
        return Err(Trap::BadConversion);
    }
    // Bounds are written as explicit comparisons so the open-vs-closed distinction is
    // visible: the i64-signed *lower* bound is closed (`>=`) because `-2^63 - 1` is
    // not representable and rounds to `-2^63`; the rest are open.
    #[allow(clippy::manual_range_contains)]
    let in_range = match (to, signed) {
        (IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (IntTy::I64, true) => f >= -9_223_372_036_854_775_808.0 && f < 9_223_372_036_854_775_808.0,
        (IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return Err(Trap::BadConversion);
    }
    // In range, so the cast is exact (truncating toward zero, no saturation).
    Ok(match (to, signed) {
        (IntTy::I32, true) => Value::I32(f as i32),
        (IntTy::I32, false) => Value::I32(f as u32 as i32),
        (IntTy::I64, true) => Value::I64(f as i64),
        (IntTy::I64, false) => Value::I64(f as u64 as i64),
    })
}

fn i_to_f(op: IToF, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        IToF::I32F32S => Value::F32(as_i32(v)? as f32),
        IToF::I32F32U => Value::F32(as_i32(v)? as u32 as f32),
        IToF::I64F32S => Value::F32(as_i64(v)? as f32),
        IToF::I64F32U => Value::F32(as_i64(v)? as u64 as f32),
        IToF::I32F64S => Value::F64(as_i32(v)? as f64),
        IToF::I32F64U => Value::F64(as_i32(v)? as u32 as f64),
        IToF::I64F64S => Value::F64(as_i64(v)? as f64),
        IToF::I64F64U => Value::F64(as_i64(v)? as u64 as f64),
    })
}

fn cast(op: CastOp, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        CastOp::Demote => Value::F32(as_f64(v)? as f32),
        CastOp::Promote => Value::F64(as_f32(v)? as f64),
        CastOp::ReinterpI32F32 => Value::F32(f32::from_bits(as_i32(v)? as u32)),
        CastOp::ReinterpF32I32 => Value::I32(as_f32(v)?.to_bits() as i32),
        CastOp::ReinterpI64F64 => Value::F64(f64::from_bits(as_i64(v)? as u64)),
        CastOp::ReinterpF64I64 => Value::I64(as_f64(v)?.to_bits() as i64),
    })
}

// ----------------------------------------------------------------------------
// Capabilities — the host-owned handle table + a deterministic mock powerbox
// (§3c index model, §3e MVP interface set). This is the reference oracle's
// stand-in for real host capabilities: deterministic, in-process, so it can be a
// differential oracle. The *security* of the model lives in `Host::resolve`
// (use-site mask + type_id + generation check → forged indices are inert).
// ----------------------------------------------------------------------------

/// MVP interface type-ids (§3e). Phase-1: a `type_id` is just a small constant a
/// handle-table entry carries and `cap.call` re-checks. (A module-level interface
/// section that globalizes ids across linked modules is deferred to §13.)
pub mod iface {
    /// `Stream` — byte stream: op 0 `read`, op 1 `write`, op 2 `close` (§3e D43).
    pub const STREAM: u32 = 0;
    /// `Exit` — lifecycle: op 0 `exit(code)` (noreturn).
    pub const EXIT: u32 = 1;
    /// `Clock` — op 0 `now(clock_id) -> i64` nanoseconds.
    pub const CLOCK: u32 = 2;
    /// `Memory` — op 0 `map`, 1 `unmap`, 2 `protect`, 3 `page_size` (§3e; real page protection —
    /// see `Mem`).
    pub const MEMORY: u32 = 3;
    /// `SharedRegion` — a host-backed memory object aliased into the window (§13). op 0
    /// `map(window_offset, region_offset, len, prot)` aliases the region's pages into the window
    /// (the same backing may be mapped at *multiple* window offsets → zero-overhead aliasing, the
    /// magic-ring-buffer primitive); op 1 `unmap(window_offset, len)` drops the alias; op 2
    /// `len() -> i64` reports the region size; op 3 `page_size() -> i64`. Granting the handle is how
    /// two domains come to share memory; `create`/`grant` (guest-minted regions, cross-domain) are a
    /// §14 follow-up — today regions are host-granted, like `Memory`.
    pub const SHARED_REGION: u32 = 4;
    /// `AddressSpace` — the §14 memory-management capability, **attenuable to a power-of-two
    /// window sub-range** `[base, base+size)`. Like `Memory` but every op is confined to the
    /// holder's sub-range (offsets are sub-range-relative, shifted by `base`): op 0 `map(off,len,prot)`,
    /// 1 `unmap(off,len)`, 2 `protect(off,len,prot)`, 3 `page_size() -> i64`, and 4
    /// **`sub(off, size_log2) -> handle`** — the **attenuation** primitive: mint a child `AddressSpace`
    /// over the power-of-two-aligned sub-range `[base+off, base+off + 2^size_log2)`, which must lie
    /// within the holder's range (a parent can only sub-allocate what it holds, §14). This is the
    /// memory half of the `Instantiator`: a guest carves a child's window from its own.
    pub const ADDRESS_SPACE: u32 = 5;
    /// `Instantiator` — the §14 nesting primitive: spawn a **child domain** confined to a
    /// power-of-two sub-window `[base, base+size)` of the holder's window (VM-in-VM). op 0
    /// `instantiate(entry, off, size_log2, fuel) -> child_handle` enqueues a child vCPU running the
    /// same module's `entry` (which returns one `i64` and takes one or two — its starter caps)
    /// confined to `[base+off, base+off+2^size_log2)` with an **attenuated** powerbox over the child's
    /// own window: an `Instantiator` (so it can recurse — confinement composes to any depth) and an
    /// `AddressSpace` (so it can manage its own pages), passed as the entry's arguments. A fuel quota
    /// caps it; returns immediately (non-blocking). op 1 `join(child_handle) -> result` parks **only
    /// the calling fiber** until that child finishes, then yields its result (siblings keep running —
    /// the child rides the same §12 executor). Holding the handle is the authority to nest (D19: a
    /// child can only get what the parent sub-allocates).
    pub const INSTANTIATOR: u32 = 6;
    /// `Yielder` — a §14 **co-fiber** child's handle back to its instantiator-parent. op 0
    /// `yield(value: i64) -> resumed: i64` suspends the child, handing `value` to the parent's
    /// `resume` (which returns it as the yield's status/value), and on the next `resume` returns the
    /// value the parent passed. The cooperative-coroutine primitive the §14 parent-virtualized-fault /
    /// lazy-paging model builds on (a child parks on a fault it cannot service; the parent supplies the
    /// page and resumes it). Granted to a coroutine child (`Instantiator.spawn_coroutine`) only.
    pub const YIELDER: u32 = 7;
    /// `Module` — a host-granted, host-**verified** module a guest may instantiate (§14). The handle
    /// confers only the authority to pass it to the `Instantiator`'s module ops (5/6/7 —
    /// `instantiate_module` / `spawn_coroutine_module` / `spawn_demand_coroutine_module`), which
    /// spawn a child domain running *that* module's code confined to a carve of the holder's window
    /// — the "plugin-in-plugin" story: a guest can only instantiate modules it was given (no ambient
    /// authority). It has no directly callable ops (`cap.call` on it is an inert `CapFault`).
    pub const MODULE: u32 = 8;
    /// §9/§12 `IoRing` — the submit/complete ring. `op 0 submit(sq_ptr, n, cq_ptr)` runs `n`
    /// deferred `cap.call`s (each a 64-byte SQE in the window) and writes their results as 32-byte
    /// CQEs, amortizing the boundary crossing — and, for *blocking* SQEs, **overlapping** them on a
    /// bounded host offload pool ([`OFFLOAD_POOL_THREADS`] threads; the §12 increment-2 win).
    pub const IO_RING: u32 = 9;
    /// §12 `Blocking` — a *mock* synchronous-only / blocking host capability (DNS-/FS-blocking-shaped)
    /// whose op 0 `work(arg) -> mix(arg)` is **window-independent and `&mut Host`-free**, so a
    /// `submit` batch can hand it to the offload pool instead of the guest's vCPU thread. Op 0 is also
    /// a perfectly ordinary synchronous `cap.call` (it then blocks the caller — the degenerate path).
    pub const BLOCKING: u32 = 10;
}

/// Negative-errno values returned by capability ops (§3e D42): `< 0` is `-errno`,
/// `>= 0` is success. Errors do **not** trap — traps stay reserved for escape/fatal.
const EFAULT: i64 = -14; // buffer not fully within the window
const EINVAL: i64 = -22; // bad op / argument
const EMFILE: i64 = -24; // handle table full — a guest-minted handle has nowhere to go (§3c)

/// A `Trap` → small status code for an `IoRing` CQE, numbered to match the JIT's `TrapKind` codes
/// (so the whole system speaks one trap-code vocabulary). `0` is reserved for success in the CQE.
fn trap_status(t: &Trap) -> i64 {
    match t {
        Trap::DivByZero => 1,
        Trap::IntOverflow => 2,
        Trap::BadConversion => 3,
        Trap::Unreachable => 4,
        Trap::IndirectCallType => 5,
        Trap::CapFault | Trap::Malformed | Trap::Exit(_) => 6, // bad/unsupported async request
        Trap::MemoryFault | Trap::StackOverflow => 8,
        Trap::FiberFault => 9,
        Trap::ThreadFault => 10,
        Trap::OutOfFuel => 11,
    }
}

/// Per-region cap on a **guest-minted** region (`AddressSpace.create_region`, §13/§14): an anti-bomb
/// ceiling so a single mint can't exhaust the host. Aggregate quota metering is §15 (D48: DoS is
/// contained by caps + the kill path, not prevented).
const MAX_MINTED_REGION: i64 = 256 << 20; // 256 MiB

/// Cap ABI `prot` bits for the `Memory` capability (§3e): the low two bits of the `i32`
/// argument. There is no `EXEC` bit — guest data is never executed as code (§3c).
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;

/// A §13 `SharedRegion`'s backing — a host-owned shared object aliased into a window at one or more
/// offsets. The reference (interpreter) backing is a plain Rust buffer ([`VecBacking`]); a flat-window
/// backend (the JIT) supplies one wrapping a real OS shared-memory object (memfd / file mapping) whose
/// [`SharedBacking::os_fd`] it `mmap`s for true hardware aliasing. Cloning the `Arc` shares the *same*
/// object, so two mappings of it alias.
///
/// `Send + Sync`: a region is shared across vCPU threads (§12) — a `Backed` page aliased into more
/// than one thread's window names the same bytes. Concurrent access is the guest's race (§12);
/// implementors serialize or use atomics as they see fit (the reference [`VecBacking`] uses a
/// `Mutex`).
pub trait SharedBacking: Send + Sync {
    /// Region size in bytes.
    fn size(&self) -> u64;
    /// Read one region-relative byte (out of range ⇒ 0).
    fn read_byte(&self, off: u64) -> u8;
    /// Write one region-relative byte (out of range ⇒ ignored). Interior-mutable: a region is shared
    /// (`Arc`), so writes go through `&self`.
    fn write_byte(&self, off: u64, b: u8);
    /// An OS shared-memory handle a flat-window backend can `mmap` for real aliasing; `None` for the
    /// pure-Rust reference backing (the interpreter models aliasing in software instead). Unix
    /// (`memfd`/`shm`); the Windows analogue is [`os_section`](SharedBacking::os_section).
    fn os_fd(&self) -> Option<i32> {
        None
    }

    /// A Windows section handle (from `CreateFileMapping`) a flat-window backend maps into the window
    /// via `MapViewOfFile3` for real aliasing — the Windows analogue of [`os_fd`](SharedBacking::os_fd).
    /// Carried as an `isize` (a `HANDLE` is pointer-sized) to keep this trait platform-clean; `None`
    /// for the pure-Rust reference backing. Only the Windows JIT path consumes it.
    fn os_section(&self) -> Option<isize> {
        None
    }
}

/// A reference to a shared region backing (see [`SharedBacking`]); cloning shares the same object.
pub type RegionBacking = Arc<dyn SharedBacking>;

/// The reference [`SharedBacking`]: a plain in-process buffer behind a `Mutex` (so it is `Send +
/// Sync` and safe to alias across vCPU threads). The interpreter models aliasing by reading/writing
/// this shared buffer through several `Backed` pages.
struct VecBacking(Mutex<Vec<u8>>);

impl VecBacking {
    /// Lock, recovering from poisoning rather than panicking (the interpreter never panics, §robust).
    fn buf(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl SharedBacking for VecBacking {
    fn size(&self) -> u64 {
        self.buf().len() as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        self.buf().get(off as usize).copied().unwrap_or(0)
    }
    fn write_byte(&self, off: u64, b: u8) {
        if let Some(s) = self.buf().get_mut(off as usize) {
            *s = b;
        }
    }
}

/// The guest window a capability handler borrows `(ptr, len)` buffers from (§7). Both
/// the interpreter's lazily-paged [`Mem`] and a JIT's flat window implement this, so a
/// single host dispatch ([`Host::cap_dispatch`]) serves both backends. **All offsets/pointers are
/// guest-relative** — the zero-based window the guest sees (a §14 child names its own `[0, size)`,
/// never its position in an ancestor's window); implementations translate to their backing. The
/// methods bounds-check `[ptr, ptr+len) ⊆ [0, size)` and return `None` (→ `-EFAULT`) otherwise.
pub trait GuestMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>>;
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()>;

    /// `Memory` capability ops (§3e): (re)commit / decommit / re-protect window pages. `offset`
    /// is page-aligned and `[offset, offset+len)` window-relative; `prot` is `READ|WRITE`. Each
    /// returns `0` or a negative errno (`-EINVAL`). The default is a success no-op — overridden
    /// by the interpreter's paged [`Mem`] (the reference semantics); a flat-window backend
    /// (e.g. a JIT) wires its own `mprotect`-backed implementation.
    fn map(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }
    fn unmap(&mut self, _offset: u64, _len: u64) -> i64 {
        0
    }
    fn protect(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }

    /// `SharedRegion` op 0 `map` (§13): alias `backing`'s `[region_off, region_off+len)` pages into
    /// the window at `[win_off, win_off+len)` with `prot`. The same `region`/`backing` mapped at two
    /// window offsets makes both ranges name the *same* bytes (zero-overhead aliasing). `0` or a
    /// negative errno. The default rejects it (`-EINVAL`): only the reference paged [`Mem`] models
    /// aliasing today; a flat-window backend wires its own shared mapping (§13 slice 2).
    fn map_region(
        &mut self,
        _win_off: u64,
        _region_off: u64,
        _len: u64,
        _prot: i32,
        _region: u32,
        _backing: RegionBacking,
    ) -> i64 {
        EINVAL
    }

    /// `Memory` op 3 `page_size() -> i64`: the host MMU page granularity this window is managed in —
    /// the unit `map`/`unmap`/`protect` round to. A guest queries it to align its own allocator to
    /// the real page (4 KiB / 16 KiB / …) and adapt, instead of assuming a fixed size. The default
    /// reports the host page; the paged [`Mem`] and the JIT's `MprotectWindow` override it with the
    /// exact value they round to, so the two backends stay in differential lockstep.
    fn page_size(&self) -> i64 {
        host_page_size() as i64
    }

    /// `SharedRegion` op 3 `page_size() -> i64`: the granularity a `SharedRegion` map aligns to —
    /// the host page on unix, the **allocation granularity** (64 KiB) on Windows, which
    /// `MapViewOfFile3` requires. Distinct from [`page_size`](GuestMem::page_size) (the protection
    /// granularity) so a guest aligns its region maps to a value that works on every backend. The
    /// default ([`host_region_granularity`]) is correct for both the paged [`Mem`] and the JIT's
    /// flat window, so the two stay in §13 lockstep without an override.
    fn region_page_size(&self) -> i64 {
        host_region_granularity() as i64
    }

    /// §9/§12 **async ring** support. Return a backend-neutral [`AsyncCounter`] for the 4-byte futex
    /// **completion counter** at `counter_addr`: an offload-pool worker atomic-increments it (the same
    /// path the backend's `wait`/`notify` value-check reads) and `notify`s its [`AsyncCounter::key`], so
    /// a vCPU parked in `wait` on the counter wakes race-free (the compare-under-lock guard). `Some`
    /// only for a normal in-window, naturally-aligned, writable page. `None` — the default — means the
    /// backend can't post async completions, so `submit_async` reports `-EINVAL` and the guest falls
    /// back to the synchronous `submit`. The reference paged [`Mem`] and the JIT's flat window both
    /// override it (each keyed to its own `wait`/`notify`: a window offset vs. an absolute address).
    fn async_counter(&self, _counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        None
    }
}

/// A `Send + Sync` handle an offload-pool worker uses to post an async-ring completion to the futex
/// **completion counter** (§9/§12). `increment` atomic-adds to the in-window counter through the same
/// path the backend's atomics take (a [`Region`] on the interpreter, a raw window write on the JIT);
/// `key` is the parking-lot key to hand the [`Host`]'s wake hook — a window offset on the interpreter
/// (the `Scheduler` key), an absolute window address on the JIT (the futex key) — each consistent with
/// that backend's `wait`/`notify`, so the worker's increment targets exactly what the parked vCPU's
/// value-check reads.
pub trait AsyncCounter: Send + Sync {
    fn increment(&self, delta: u64);
    fn key(&self) -> u64;
}

/// §4/§7 a JIT cap-path window **page map**: page index → state code (the flat-window backend, e.g.
/// `svm_run`, owns the encoding; absent ⇒ region default). Shared + persistent across a run's
/// `cap.call`s (see [`Host::cap_window_pages`]) so a guest-grown page stays borrowable.
pub type CapPageMap = Arc<Mutex<BTreeMap<u64, u8>>>;

/// A [`GuestMem`] over a flat, contiguous window slice — the JIT's representation. The
/// slice may include trailing guard bytes; `size` is the *logical* window so the §7
/// bounds check matches the interpreter exactly.
pub struct WindowMem<'a> {
    window: &'a mut [u8],
    size: u64,
}

impl<'a> WindowMem<'a> {
    pub fn new(window: &'a mut [u8], size: u64) -> WindowMem<'a> {
        WindowMem { window, size }
    }
}

impl GuestMem for WindowMem<'_> {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        if end > self.size {
            return None;
        }
        Some(self.window[ptr as usize..end as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let end = ptr.checked_add(data.len() as u64)?;
        if end > self.size {
            return None;
        }
        self.window[ptr as usize..end as usize].copy_from_slice(data);
        Some(())
    }
}

/// Which standard stream a `Stream` handle is bound to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamRole {
    In,
    Out,
    Err,
}

/// The host-side object a handle-table entry dispatches to — the mock equivalent of
/// §3c's `(methods, object)`. The guest never names or writes this (it lives in host
/// memory); it is selected only by a *granted* handle index.
#[derive(Clone, Copy, Debug)]
enum Binding {
    Stream(StreamRole),
    Exit,
    Clock,
    Memory,
    /// A §13 `SharedRegion` handle, carrying the index of its backing in [`Host::regions`]. The
    /// backing (not the index) is the shared object; mapping it at several window offsets aliases.
    SharedRegion(u32),
    /// A §14 `AddressSpace` handle attenuated to the power-of-two window sub-range `[base, base+size)`
    /// in the **holder's own (guest-relative) coordinates** — a child's full-window grant is
    /// `[0, its size)` regardless of where its window sits in an ancestor's. Every op is confined to
    /// it; `sub` mints a further-attenuated child. The bounds live in the host-owned slot — the guest
    /// names only the forgeable handle.
    AddressSpace {
        base: u64,
        size: u64,
    },
    /// A §14 `Instantiator` handle conferring authority to spawn children confined to the window
    /// sub-range `[base, base+size)` in the **holder's own (guest-relative) coordinates**. The eval
    /// loop (not the generic dispatch) services it — spawning needs executor access — translating to
    /// backing-absolute via the holder's window base, so nesting composes at any depth.
    Instantiator {
        base: u64,
        size: u64,
    },
    /// A §14 `Yielder` handle a co-fiber child holds to suspend back to its instantiator-parent. The
    /// eval loop services it (it must yield the running coroutine's continuation, which the generic
    /// dispatch can't); a forged/wrong handle resolves nowhere and is an inert `CapFault`.
    Yielder,
    /// A §14 `Module` handle, carrying the index of its grant in [`Host::modules`]. Confers only the
    /// authority to instantiate (the Instantiator's module ops, serviced by the eval loop / nesting
    /// runtime); the generic dispatch treats any `cap.call` on it as an inert `CapFault`.
    Module(u32),
    /// A §9/§12 `IoRing` handle: authority to `submit` a batch of deferred `cap.call`s
    /// (io_uring-shaped), carrying the index of its [`RingState`] in [`Host::rings`] (the async-path
    /// completion buffer; the synchronous `submit` doesn't use it). The SQ/CQ ring buffers live in the
    /// guest window; the ops get their pointers as args.
    IoRing(u32),
    /// A §12 `Blocking` handle, carrying the index of its [`AsyncState`] in [`Host::blockings`] — a
    /// mock synchronous-only/blocking op the offload pool can overlap. Out-of-line (an index, not the
    /// `Arc`) so `Binding` stays `Copy`, like [`Binding::SharedRegion`]/[`Binding::Module`].
    Blocking(u32),
}

/// One handle-table slot (§3c): host-owned, guest-unwritable. `generation` is
/// per-slot and only advances on (re)grant, so a closed handle's value can never
/// alias a later grant of the same slot (ABA-safe use-after-close detection, D37).
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    generation: u32,
    entry: Option<Binding>,
    type_id: u32,
}

/// `log2` of the handle-table capacity. A handle value packs `(generation, slot)`:
/// `slot = h & (cap-1)`, `generation = h >> CAP_LOG2`.
const CAP_LOG2: u32 = 8;
const CAP: usize = 1 << CAP_LOG2;

/// Worker-thread count of the host **bounded blocking-offload pool** (§12 "Keeping cores busy under
/// blocking", path 2 — the io_uring increment-2 path). At most this many *blocking* SQEs run
/// concurrently; the `(K+1)`th queues — so the OS-thread cost of a `submit` batch is bounded by `K`,
/// never by the number of deferred ops. The guest's own vCPU thread is **not** multiplied (it parks
/// on the one `submit` while the pool absorbs the blocking) — the "0 blocked vCPU threads" win.
pub const OFFLOAD_POOL_THREADS: usize = 4;

/// Shared, thread-safe state behind a [`iface::BLOCKING`] capability — a *mock* synchronous-only host
/// op used to exercise the offload pool. Its `run` is **window-independent and `&mut Host`-free** (a
/// pure function of its argument plus this `Send + Sync` state), which is exactly the property that
/// lets a `submit` batch run it on the pool instead of the guest's vCPU thread. The result is
/// deterministic ⇒ both backends agree (the §18 oracle); the `active`/`max_active` counters let a
/// test *prove* a batch genuinely overlapped on the pool.
pub struct AsyncState {
    /// How long each op blocks before returning — the "synchronous blocking" the pool absorbs.
    /// `Duration::ZERO` in production (a pure compute); a test sets it to make the blocking real.
    block_for: Duration,
    /// Optional rendezvous: when set, every concurrent op waits here before completing, so a batch of
    /// exactly `width` ops on a `≥ width`-thread pool **deterministically** co-resides
    /// (`max_active == width`) without depending on sleep timing. `None` in production. A *direct*
    /// (non-batched) `cap.call` on a rendezvous-configured handle would block forever — it is a
    /// batch-overlap test fixture only.
    rendezvous: Option<Arc<Barrier>>,
    /// Ops currently in-flight (bumped on entry, dropped on completion).
    active: AtomicUsize,
    /// High-water mark of `active` across this `AsyncState`'s lifetime — the realized concurrency,
    /// read back via [`AsyncState::max_active`] to confirm a batch overlapped on `K` threads.
    max_active: AtomicUsize,
}

impl AsyncState {
    /// Run one blocking op: account the in-flight concurrency, (optionally) rendezvous + block, then
    /// return the deterministic transform of `arg`. Called either inline (a direct `cap.call`, on the
    /// caller's thread) or on an offload-pool worker (a batched `submit`) — same result either way.
    fn run(&self, arg: i64) -> i64 {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        if let Some(b) = &self.rendezvous {
            b.wait();
        }
        if !self.block_for.is_zero() {
            std::thread::sleep(self.block_for);
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        Self::mix(arg)
    }

    /// A deterministic, non-trivial pure transform (one Knuth LCG step) — identical on every backend
    /// and thread, so a batch's CQE results are reproducible (and a divergence would show).
    fn mix(arg: i64) -> i64 {
        arg.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    /// The peak realized concurrency — a test reads this after a batched `submit` to confirm the pool
    /// overlapped the blocking ops (`== min(batch, OFFLOAD_POOL_THREADS)`).
    pub fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
}

/// A job handed to the offload pool: a self-contained closure that writes its own result (it captures
/// the destination), so completion *order* is irrelevant — the data it leaves is deterministic.
type OffloadJob = Box<dyn FnOnce() + Send + 'static>;

/// The **bounded blocking-offload pool** (§12 path 2): [`OFFLOAD_POOL_THREADS`] long-lived workers
/// that run window-independent blocking SQEs *off* the guest's vCPU thread. A `submit` of `n` blocking
/// ops costs `K` OS threads regardless of `n` (waves of `K`). Each worker owns its **own** channel —
/// a single shared `Mutex<Receiver>` would serialize the blocking `recv`s and defeat the overlap.
struct OffloadPool {
    /// Per-worker job channels; a batch is round-robined across them.
    txs: Vec<std::sync::mpsc::Sender<OffloadJob>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Jobs dispatched but not yet finished — `(count, condvar)`. [`OffloadPool::dispatch`] (the async
    /// path) returns without waiting, so [`OffloadPool::quiesce`] uses this to drain in-flight work at
    /// run end before the window's `Arc<Region>` (which a late job may still write) is read back.
    inflight: Arc<(Mutex<usize>, Condvar)>,
    next: usize,
}

impl OffloadPool {
    fn new(k: usize) -> OffloadPool {
        let mut txs = Vec::with_capacity(k);
        let mut workers = Vec::with_capacity(k);
        for _ in 0..k {
            let (tx, rx) = std::sync::mpsc::channel::<OffloadJob>();
            txs.push(tx);
            workers.push(std::thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            }));
        }
        OffloadPool {
            txs,
            workers,
            inflight: Arc::new((Mutex::new(0), Condvar::new())),
            next: 0,
        }
    }

    /// **Async dispatch** (the increment-3 path): round-robin `jobs` to the workers and return
    /// **immediately**. Each job is wrapped to decrement the in-flight count + notify on completion;
    /// the job itself posts its own completion (host-side result + futex counter + `notify`). The
    /// guest's vCPU parks via the futex `wait` rather than blocking here — the whole point of async.
    fn dispatch(&mut self, jobs: Vec<OffloadJob>) {
        if jobs.is_empty() {
            return;
        }
        {
            let (m, _) = &*self.inflight;
            *m.lock().unwrap() += jobs.len();
        }
        for job in jobs {
            let inflight = Arc::clone(&self.inflight);
            let wrapped: OffloadJob = Box::new(move || {
                job();
                let (m, c) = &*inflight;
                let mut g = m.lock().unwrap();
                *g -= 1;
                if *g == 0 {
                    c.notify_all();
                }
            });
            let w = self.next % self.txs.len();
            self.next = self.next.wrapping_add(1);
            self.txs[w].send(wrapped).expect("offload worker vanished");
        }
    }

    /// Block until every dispatched async job has finished. Called at run end so no worker still holds
    /// (and might still write) the window's `Arc<Region>` after the caller reads the final memory back.
    fn quiesce(&self) {
        let (m, c) = &*self.inflight;
        let mut g = m.lock().unwrap();
        while *g > 0 {
            g = c.wait(g).unwrap();
        }
    }

    /// Round-robin `jobs` across the workers and **block until all complete**. Each job writes its
    /// result through its own captured destination, so the caller reads results back by index after
    /// this returns. This is the synchronous-submit MVP: one boundary crossing, `K`-way overlap,
    /// then a single reap (fiber-parking / async resume is increment 3).
    fn run_batch(&self, jobs: Vec<OffloadJob>) {
        let n = jobs.len();
        if n == 0 {
            return;
        }
        let done = Arc::new((Mutex::new(0usize), Condvar::new()));
        for (i, job) in jobs.into_iter().enumerate() {
            let done = Arc::clone(&done);
            let wrapped: OffloadJob = Box::new(move || {
                job();
                let (m, c) = &*done;
                *m.lock().unwrap() += 1;
                c.notify_all();
            });
            // `send` only fails if a worker thread is gone — a host bug, not a guest-reachable path,
            // and the wait below would then hang, so surface it loudly.
            self.txs[i % self.txs.len()]
                .send(wrapped)
                .expect("offload worker vanished");
        }
        let (m, c) = &*done;
        let mut g = m.lock().unwrap();
        while *g < n {
            g = c.wait(g).unwrap();
        }
    }
}

impl Drop for OffloadPool {
    fn drop(&mut self) {
        // Dropping the senders closes each worker's channel → its `recv` returns `Err` → it exits.
        self.txs.clear();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// §9/§12 async-ring per-handle state: completions posted by offload workers (or by inline ops) during
/// a `submit_async`, awaiting the guest's `reap` to flush them into the window. `Send + Sync` so a pool
/// worker pushes from its own thread; the guest reaps on its vCPU thread. The futex completion counter
/// lives in the *window* (so the guest can `wait` on it); this holds only the CQE payloads.
#[derive(Default)]
struct RingState {
    /// Ready completions `(user_data, result, status)`, FIFO — pushed by workers/inline, popped by reap.
    completed: Mutex<VecDeque<(i64, i64, i64)>>,
}

/// The interpreter's [`AsyncCounter`]: the futex completion counter is a normal anonymous window page,
/// so a worker increments it via the shared [`Region`] (the same real-atomic path cross-vCPU atomics
/// take) and the parking key is the window-relative offset (the `Scheduler`'s parking-lot key).
struct RegionCounter {
    region: Arc<Region>,
    off: u64,
}

impl AsyncCounter for RegionCounter {
    fn increment(&self, delta: u64) {
        self.region.atomic_rmw(self.off, 4, RmwOp::Add, delta);
    }
    fn key(&self) -> u64 {
        self.off
    }
}

/// The host: the **host-owned handle table** (the powerbox) plus deterministic mock
/// capability state (captured stdio, a monotonic clock). Construct with [`Host::new`],
/// `grant_*` the initial capabilities, then pass to [`run_with_host`]; afterwards read
/// back `stdout`/`stderr`. Deterministic by design so it serves as a §18 oracle.
pub struct Host {
    table: Vec<Slot>, // CAP slots, host-owned
    /// Bytes a `Stream{In}` handle's `read` draws from.
    pub stdin: Vec<u8>,
    stdin_pos: usize,
    /// Bytes written by `Stream{Out}` / `Stream{Err}` `write`s.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// Monotonic nanosecond counter; each `Clock.now` returns it then advances by one,
    /// so reads are deterministic and strictly increasing.
    pub clock_ns: i64,
    /// §13 `SharedRegion` backings, indexed by the id a [`Binding::SharedRegion`] carries. Each is a
    /// shared host buffer; aliasing a region into several window offsets clones this `Rc`.
    regions: Vec<RegionBacking>,
    /// §14 instantiable **modules**, indexed by the id a [`Binding::Module`] carries — host-verified
    /// code a guest holding the handle may spawn as a child domain (`Arc`s so a spawned child shares,
    /// not copies). Append-only for the life of the `Host`, so raw views handed to the JIT's nesting
    /// runtime ([`Host::resolve_module_parts`]) stay valid for the whole run.
    modules: Vec<ModuleGrant>,
    /// The backing factory for **guest-minted** §13/§14 regions (`AddressSpace.create_region`).
    /// `None` (the default) mints the pure-Rust reference [`VecBacking`]; a flat-window embedder
    /// installs an OS-shared-memory factory ([`Host::set_region_factory`], e.g.
    /// `svm_run::new_shared_region`) so a JIT guest can `map` what it mints for real aliasing.
    region_factory: Option<fn(usize) -> RegionBacking>,
    /// §12 `Blocking` capability backings, indexed by the id a [`Binding::Blocking`] carries. Each is
    /// a `Send + Sync` [`AsyncState`] a `submit` batch can run on the offload pool.
    blockings: Vec<Arc<AsyncState>>,
    /// The §12 bounded blocking-offload pool, created lazily on the first batched `submit` that has a
    /// blocking SQE (so a `Host` that never offloads spawns no threads). Dropping it joins the
    /// workers ([`OffloadPool`]'s `Drop`).
    pool: Option<OffloadPool>,
    /// §9/§12 async-ring per-handle state, indexed by the id a [`Binding::IoRing`] carries — where a
    /// `submit_async` posts completions for the guest's `reap`.
    rings: Vec<Arc<RingState>>,
    /// §9/§12 the **async-completion `notify`** hook: an offload worker calls this (with the confined
    /// futex counter key) to wake the vCPU parked in `wait` on that key — i.e. an I/O completion *is* a
    /// futex notify (DESIGN §12). Installed per run by the executor that owns the wake mechanism
    /// (`drive` wires it to the M:N `Scheduler::notify`); `None` ⇒ no async support, so `submit_async`
    /// `-EINVAL`s and the guest falls back to the synchronous `submit`.
    async_notify: Option<Arc<dyn Fn(u64, u32) + Send + Sync>>,
    /// §4/§7 the **JIT cap-path window page map**, keyed by window base. The JIT's `cap_thunk` rebuilds
    /// its window view per `cap.call`, so without a persistent home a guest-*grown* heap page (committed
    /// via the Memory cap in an earlier call) would read back as unmapped and a cap-buffer borrow of it
    /// would fail-closed. Persisting it here (the per-run `Host` is the only state `cap_thunk` reaches)
    /// mirrors how the interpreter's `Mem` keeps its page map across calls. Page index → state code
    /// (`svm_run` owns the encoding); absent ⇒ region default. Reset when a new window base appears.
    cap_pages: Option<(usize, CapPageMap)>,
}

/// One §14 module grant: the verified module's functions, declared window size, and data segments —
/// what spawning a child domain of it needs.
struct ModuleGrant {
    funcs: Arc<[Func]>,
    memory_log2: Option<u8>,
    data: Arc<[Data]>,
}

impl Default for Host {
    fn default() -> Host {
        Host::new()
    }
}

impl Host {
    pub fn new() -> Host {
        Host {
            table: vec![Slot::default(); CAP],
            stdin: Vec::new(),
            stdin_pos: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            clock_ns: 0,
            regions: Vec::new(),
            modules: Vec::new(),
            region_factory: None,
            blockings: Vec::new(),
            pool: None,
            rings: Vec::new(),
            async_notify: None,
            cap_pages: None,
        }
    }

    /// §4/§7 the JIT cap-path window page map (see [`Host::cap_pages`]) for window `base`, persistent
    /// across this run's `cap.call`s so a guest-grown heap page stays borrowable. Returns a fresh empty
    /// map when the base changes (a new window / run reusing this `Host`), else the existing one.
    pub fn cap_window_pages(&mut self, base: usize) -> CapPageMap {
        match &self.cap_pages {
            Some((b, m)) if *b == base => Arc::clone(m),
            _ => {
                let m = Arc::new(Mutex::new(BTreeMap::new()));
                self.cap_pages = Some((base, Arc::clone(&m)));
                m
            }
        }
    }

    /// Install the §9/§12 async-completion `notify` hook (the executor that owns the wake mechanism
    /// wires it at run start; see [`Host::async_notify`]). The interp's `drive` calls this with the M:N
    /// `Scheduler::notify`; the JIT wires its futex via the same seam (`svm_jit::AsyncHostHooks`).
    /// Cleared at run end to drop the closure's executor reference.
    pub fn set_async_notify(&mut self, f: Arc<dyn Fn(u64, u32) + Send + Sync>) {
        self.async_notify = Some(f);
    }
    pub fn clear_async_notify(&mut self) {
        self.async_notify = None;
    }
    /// Drain any in-flight offload-pool jobs (run end), so no worker still holds the window backing (or
    /// the JIT's window/`Domain` pointers) when the caller frees them.
    pub fn quiesce_pool(&self) {
        if let Some(p) = &self.pool {
            p.quiesce();
        }
    }

    /// Install a host binding in a free slot and return the guest handle — a forgeable
    /// `i32` index encoding `(generation, slot)`. This is how the powerbox (and, later,
    /// attenuation) hands authority to the guest (§3c). Panics only if the table is
    /// full (a host bug, not reachable from guest code).
    /// Fallible grant: claim a free handle-table slot for `binding`, or `None` if the table is full
    /// (all `CAP` slots live). **Guest-minting** ops (`AddressSpace.sub`, `create_region`, the
    /// cross-domain `SharedRegion.grant`) must use this and surface `None` as `-EMFILE` — a guest can
    /// call them in a loop, and a panic here would unwind across the JIT's `extern "C"` cap thunk and
    /// abort the host (a guest must never crash the host; §5). Host-side powerbox setup uses the
    /// infallible [`Host::grant`] (it grants a bounded few into a fresh table at instantiation).
    fn try_grant(&mut self, type_id: u32, binding: Binding) -> Option<i32> {
        let slot = self.table.iter().position(|s| s.entry.is_none())?;
        let s = &mut self.table[slot];
        s.generation = s.generation.wrapping_add(1); // advance per (re)grant (ABA-safe)
        s.entry = Some(binding);
        s.type_id = type_id;
        Some(((s.generation << CAP_LOG2) | slot as u32) as i32)
    }

    /// Infallible grant for **host-controlled** powerbox setup (`grant_stream`/`grant_memory`/… and
    /// the `grant_*` embedder APIs): the host grants a bounded handful into a fresh `CAP`-slot table,
    /// so the table cannot be full. Never call this on a **guest-reachable** path — use
    /// [`Host::try_grant`] there (a guest can exhaust the table; see its docs).
    fn grant(&mut self, type_id: u32, binding: Binding) -> i32 {
        self.try_grant(type_id, binding)
            .expect("handle table full during host powerbox setup (bounded by construction)")
    }

    /// Grant a `Stream` capability bound to `role` (a powerbox stdio grant, §3e).
    pub fn grant_stream(&mut self, role: StreamRole) -> i32 {
        self.grant(iface::STREAM, Binding::Stream(role))
    }
    pub fn grant_exit(&mut self) -> i32 {
        self.grant(iface::EXIT, Binding::Exit)
    }
    pub fn grant_clock(&mut self) -> i32 {
        self.grant(iface::CLOCK, Binding::Clock)
    }
    pub fn grant_memory(&mut self) -> i32 {
        self.grant(iface::MEMORY, Binding::Memory)
    }
    /// Grant a §9/§12 `IoRing` capability — authority to `submit` batched/deferred `cap.call`s
    /// (synchronously via op 0, or asynchronously via op 1 `submit_async` + op 2 `reap`).
    pub fn grant_io_ring(&mut self) -> i32 {
        let idx = self.rings.len() as u32;
        self.rings.push(Arc::new(RingState::default()));
        self.grant(iface::IO_RING, Binding::IoRing(idx))
    }
    /// Grant a §12 `Blocking` capability — a mock synchronous/blocking host op the offload pool can
    /// overlap. `block_for` is how long each op blocks (`Duration::ZERO` for a pure compute);
    /// `rendezvous` (test-only) installs a width-`w` [`Barrier`] so a batch of exactly `w` ops on a
    /// `≥ w`-thread pool deterministically co-resides (proving overlap without timing).
    pub fn grant_blocking(&mut self, block_for: Duration, rendezvous: Option<usize>) -> i32 {
        let state = Arc::new(AsyncState {
            block_for,
            rendezvous: rendezvous.map(|w| Arc::new(Barrier::new(w))),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let idx = self.blockings.len() as u32;
        self.blockings.push(state);
        self.grant(iface::BLOCKING, Binding::Blocking(idx))
    }
    /// Read back the [`AsyncState`] behind a granted `Blocking` handle (a test inspects `max_active`
    /// to confirm a batched `submit` overlapped on the pool). `None` if the handle isn't a `Blocking`.
    pub fn blocking_state(&self, handle: i32) -> Option<Arc<AsyncState>> {
        match self.resolve(handle, iface::BLOCKING) {
            Ok(Binding::Blocking(idx)) => self.blockings.get(idx as usize).cloned(),
            _ => None,
        }
    }

    /// Grant a §14 `AddressSpace` capability over the window sub-range `[base, base+size)` (§14). The
    /// root grant is normally the whole window (`base = 0`, `size` the window size); the guest then
    /// `sub`-attenuates it to carve children. `size` must be a power of two and `base` a multiple of
    /// it (so the range and every sub-range are power-of-two aligned, §4/D19) — the caller's
    /// contract, mirroring how the host lays out windows.
    pub fn grant_address_space(&mut self, base: u64, size: u64) -> i32 {
        self.grant(iface::ADDRESS_SPACE, Binding::AddressSpace { base, size })
    }

    /// Grant a §14 `Instantiator` capability over the window sub-range `[base, base+size)` — the
    /// authority to spawn children (`instantiate`/`join`) confined to power-of-two sub-windows of it
    /// (§14). Like `grant_address_space`, `size` must be a power of two and `base` a multiple of it.
    pub fn grant_instantiator(&mut self, base: u64, size: u64) -> i32 {
        self.grant(iface::INSTANTIATOR, Binding::Instantiator { base, size })
    }

    /// Resolve a handle as an `Instantiator` (§14) and return its `(base, size)` sub-range, or a
    /// `CapFault` for a forged / closed / wrong-type handle. Used by the eval loop, which services
    /// `instantiate`/`join` itself (the generic dispatch can't reach the executor).
    fn resolve_instantiator(&self, handle: i32) -> Result<(u64, u64), Trap> {
        match self.resolve(handle, iface::INSTANTIATOR)? {
            Binding::Instantiator { base, size } => Ok((base, size)),
            _ => Err(Trap::CapFault),
        }
    }

    /// Grant a §14 `Yielder` capability (the co-fiber child's handle back to its parent). Used by the
    /// eval loop when standing up a coroutine child; not a powerbox-level grant.
    fn grant_yielder(&mut self) -> i32 {
        self.grant(iface::YIELDER, Binding::Yielder)
    }

    /// Confirm a handle resolves to *this* domain's `Yielder` (§14 co-fiber); a forged/wrong handle is
    /// a `CapFault`. The eval loop calls this before yielding the running coroutine's continuation.
    fn resolve_yielder(&self, handle: i32) -> Result<(), Trap> {
        match self.resolve(handle, iface::YIELDER)? {
            Binding::Yielder => Ok(()),
            _ => Err(Trap::CapFault),
        }
    }

    /// Grant a §14 **`Module` capability** over `m` — the authority to instantiate it as a child
    /// domain via the `Instantiator`'s module ops (the "plugin" grant). **`m` must already be
    /// verified** (`svm_verify::verify_module`): like every run entry, the host is trusted to grant
    /// only verifier-passing modules — a guest can never inject code, only spawn what it was given.
    pub fn grant_module(&mut self, m: &Module) -> i32 {
        let id = self.modules.len() as u32;
        self.modules.push(ModuleGrant {
            funcs: m.funcs.clone().into(),
            memory_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone().into(),
        });
        self.grant(iface::MODULE, Binding::Module(id))
    }

    /// Resolve a handle as a §14 `Module` grant — the eval loop's lookup for the Instantiator's
    /// module ops. A forged / closed / wrong-type handle is a `CapFault`.
    fn resolve_module(&self, handle: i32) -> Result<&ModuleGrant, Trap> {
        match self.resolve(handle, iface::MODULE)? {
            Binding::Module(id) => self.modules.get(id as usize).ok_or(Trap::CapFault),
            _ => Err(Trap::CapFault),
        }
    }

    /// Resolve a §14 `Module` handle to **raw views** of its grant — the bridge the JIT's nesting
    /// runtime uses (via `svm-run`'s `module_resolver` callback; `svm-jit` cannot name `Host`).
    /// `None` for a forged/closed/wrong-type handle. The returned pointers borrow [`Host::modules`]
    /// (append-only), so they stay valid for as long as this `Host` lives — which outlives the run,
    /// the same lifetime contract as the `cap.call` ctx itself. `memory_log2` is `-1` when the
    /// module declares no memory. Host-side callers only; never reachable from a guest `cap.call`
    /// (the generic dispatch on a `Module` handle is an inert `CapFault`), so no host address ever
    /// leaks into a guest-readable value.
    #[allow(clippy::type_complexity)]
    pub fn resolve_module_parts(
        &self,
        handle: i32,
    ) -> Option<(*const Func, usize, i32, *const Data, usize)> {
        let g = self.resolve_module(handle).ok()?;
        Some((
            g.funcs.as_ptr(),
            g.funcs.len(),
            g.memory_log2.map_or(-1, |l| l as i32),
            g.data.as_ptr(),
            g.data.len(),
        ))
    }

    /// Grant a §13 `SharedRegion` capability backed by a fresh `len`-byte zero-filled host buffer,
    /// returning its handle. The guest `map`s it into its window (op 0) — at one or more offsets — to
    /// access the shared bytes as ordinary masked loads/stores. (Guest-minted regions and
    /// cross-domain `grant` are a §14 follow-up; this models the host↔guest data plane.)
    pub fn grant_shared_region(&mut self, len: usize) -> i32 {
        self.grant_shared_region_backed(Arc::new(VecBacking(Mutex::new(vec![0u8; len]))))
    }

    /// Grant a §13 `SharedRegion` over a caller-supplied [`SharedBacking`] — how a flat-window
    /// backend installs a region whose `os_fd` it can `mmap` for real hardware aliasing (the JIT
    /// side of the §13 differential). The pure-Rust [`grant_shared_region`] is the common case.
    pub fn grant_shared_region_backed(&mut self, backing: RegionBacking) -> i32 {
        let id = self.regions.len() as u32;
        self.regions.push(backing);
        self.grant(iface::SHARED_REGION, Binding::SharedRegion(id))
    }

    /// Fallible [`grant_shared_region_backed`] for **guest-minting** paths (`create_region`, the
    /// cross-domain `grant`): `None` when the handle table is full (so the caller can return
    /// `-EMFILE` instead of panicking). Checks for a free slot **before** registering the backing, so
    /// a full table leaves `regions` untouched (no leaked backing).
    pub fn try_grant_shared_region_backed(&mut self, backing: RegionBacking) -> Option<i32> {
        if self.table.iter().all(|s| s.entry.is_some()) {
            return None; // table full — don't register a backing we can't hand out
        }
        let id = self.regions.len() as u32;
        self.regions.push(backing);
        self.try_grant(iface::SHARED_REGION, Binding::SharedRegion(id))
    }

    /// Install the backing factory for **guest-minted** regions (`AddressSpace.create_region`,
    /// §13/§14). A flat-window embedder passes an OS-shared-memory factory (e.g.
    /// `svm_run::new_shared_region`) so a JIT guest can `map` what it mints; without one, mints use
    /// the pure-Rust reference [`VecBacking`] (fine for the interpreter, unmappable by the JIT).
    pub fn set_region_factory(&mut self, f: fn(usize) -> RegionBacking) {
        self.region_factory = Some(f);
    }

    /// Resolve a handle as a §13 `SharedRegion` and return its backing (an `Arc` clone — the same
    /// shared object). Used by the eval loop's cross-domain `grant` (SharedRegion op 4); a forged /
    /// closed / wrong-type handle is a `CapFault`.
    fn resolve_region(&self, handle: i32) -> Result<RegionBacking, Trap> {
        match self.resolve(handle, iface::SHARED_REGION)? {
            Binding::SharedRegion(id) => {
                self.regions.get(id as usize).cloned().ok_or(Trap::CapFault)
            }
            _ => Err(Trap::CapFault),
        }
    }

    /// Close a handle (§3c): free the slot but keep its generation, so the old handle
    /// value is now a dead generation and any later `cap.call` on it traps (D37).
    pub fn close(&mut self, handle: i32) {
        let slot = (handle as u32 as usize) & (CAP - 1);
        self.table[slot].entry = None;
    }

    /// Resolve a handle at a `cap.call` use site (§3c) — **the security hinge**: mask
    /// the index into the host-owned table (never branch), then re-check the entry's
    /// interface `type_id` and `generation`. A forged / closed / wrong-type index is
    /// inert: it faults, or at worst selects one of *this domain's own* granted
    /// `type_id` capabilities. The guest never supplies the binding.
    fn resolve(&self, handle: i32, type_id: u32) -> Result<Binding, Trap> {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1); // mask, not branch (Spectre-v1 safe)
        let gen = h >> CAP_LOG2;
        let s = &self.table[slot];
        match s.entry {
            Some(b) if s.type_id == type_id && s.generation == gen => Ok(b),
            _ => Err(Trap::CapFault),
        }
    }

    /// Dispatch a `cap.call` (§3c): resolve the handle, then run the mock operation.
    /// Returns the op's result values (negative-errno encoded in an `i64` for the
    /// fallible ops, §3e D42), or a `Trap` for escape/exit. `mem` backs buffer args.
    /// Dispatch a `cap.call` (§3c): resolve the handle in the host-owned table, then run
    /// the bound capability op. Public and **slot-based** (`i64` per scalar; `i32` in
    /// the low bits) so both backends drive the same handlers without per-arg type tags
    /// — the interpreter converts its `Value`s, a JIT passes its slots directly. `mem`
    /// is `None` when the module declares no memory (buffer ops then return `-EFAULT`).
    pub fn cap_dispatch_slots(
        &mut self,
        type_id: u32,
        op: u32,
        handle: i32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        match self.resolve(handle, type_id)? {
            Binding::Stream(role) => self.stream_op(role, op, args, mem),
            Binding::Exit => {
                // op 0: exit(code: i32) — noreturn. Propagate as a (non-error) trap.
                let code = *args.first().ok_or(Trap::Malformed)? as i32;
                Err(Trap::Exit(code))
            }
            Binding::Clock => {
                // op 0: now(clock_id) -> i64 nanoseconds (deterministic, increasing).
                let now = self.clock_ns;
                self.clock_ns = self.clock_ns.wrapping_add(1);
                Ok(vec![now])
            }
            Binding::Memory => {
                // map(off,len,prot) / unmap(off,len) / protect(off,len,prot) on the window's
                // pages (§3e). With no window there is nothing to address (-EINVAL); the effect
                // is applied to whichever backend's memory `mem` wraps (interp `Mem` here, a
                // JIT's flat window via its own impl), keeping the two in differential lockstep.
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                let off = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let prot = *args.get(2).unwrap_or(&0) as i32;
                Ok(vec![match op {
                    0 => mem.map(off, len, prot),
                    1 => mem.unmap(off, len),
                    2 => mem.protect(off, len, prot),
                    3 => mem.page_size(),
                    _ => EINVAL,
                }])
            }
            Binding::SharedRegion(region) => {
                // §13: alias the host-backed region into the window. `map` (op 0) at several offsets
                // aliases the same bytes; loads/stores then go through the ordinary masked path.
                let Some(backing) = self.regions.get(region as usize).cloned() else {
                    return Ok(vec![EINVAL]);
                };
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                Ok(vec![match op {
                    0 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let region_off = *args.get(1).unwrap_or(&0) as u64;
                        let len = *args.get(2).unwrap_or(&0) as u64;
                        let prot = *args.get(3).unwrap_or(&0) as i32;
                        mem.map_region(win_off, region_off, len, prot, region, backing)
                    }
                    1 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let len = *args.get(1).unwrap_or(&0) as u64;
                        mem.unmap(win_off, len)
                    }
                    2 => backing.size() as i64,
                    3 => mem.region_page_size(),
                    _ => EINVAL,
                }])
            }
            Binding::AddressSpace { base, size } => {
                // §14: every op is confined to this capability's sub-range `[base, base+size)`. Offsets
                // are sub-range-relative; the handler bounds them and shifts by `base` into the window,
                // so a holder can never reach a byte outside its grant — the memory authority a child
                // gets from the `Instantiator`. `sub` (op 4) is **attenuation**: mint a child range.
                if op == 4 {
                    // sub(off, size_log2) -> child handle (attenuation). Mint an AddressSpace over the
                    // power-of-two-aligned `[base+off, base+off+child)` ⊆ `[base, base+size)`.
                    let off = *args.first().unwrap_or(&0) as u64;
                    let size_log2 = *args.get(1).unwrap_or(&-1);
                    if !(0..64).contains(&size_log2) {
                        return Ok(vec![EINVAL]);
                    }
                    let child = 1u64 << size_log2;
                    // child fits, `off` is child-aligned (power-of-two sub-window, D19), and the whole
                    // child range lies within this holder's range — "sub-allocate only what you hold".
                    let fits = child <= size
                        && off & (child - 1) == 0
                        && off.checked_add(child).is_some_and(|end| end <= size);
                    if !fits {
                        return Ok(vec![EINVAL]);
                    }
                    // Guest-minting: a full handle table yields -EMFILE, never a panic (§3c / audit #1).
                    return Ok(vec![match self.try_grant(
                        iface::ADDRESS_SPACE,
                        Binding::AddressSpace {
                            base: base + off,
                            size: child,
                        },
                    ) {
                        Some(h) => h as i64,
                        None => EMFILE,
                    }]);
                }
                if op == 5 {
                    // create_region(len) -> region handle — a **guest-minted** §13/§14 `SharedRegion`
                    // (the cross-domain data plane's `create`): the memory-management authority mints
                    // a fresh zero-filled shareable region and gets its handle, to `map` into its own
                    // window and/or `grant` into a child domain (SharedRegion op 4). Backing comes
                    // from the embedder's factory (OS shared memory under the JIT) or the reference
                    // `VecBacking`. Capped per-region (anti-bomb); real quota metering is §15 — DoS
                    // is contained, not prevented (D48).
                    let len = *args.first().unwrap_or(&0);
                    if len <= 0 || len > MAX_MINTED_REGION {
                        return Ok(vec![EINVAL]);
                    }
                    let backing = match self.region_factory {
                        Some(f) => f(len as usize),
                        None => Arc::new(VecBacking(Mutex::new(vec![0u8; len as usize]))),
                    };
                    // Guest-minting: a full handle table yields -EMFILE, never a panic (§3c / audit #1).
                    return Ok(vec![self
                        .try_grant_shared_region_backed(backing)
                        .map_or(EMFILE, |h| h as i64)]);
                }
                // map/unmap/protect/page_size — same shapes as `Memory`, but bounded to `[0, size)`
                // and shifted by `base` (a buffer/range straddling the sub-range boundary is -EINVAL).
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                if op == 3 {
                    return Ok(vec![mem.page_size()]);
                }
                let off = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let prot = *args.get(2).unwrap_or(&0) as i32;
                // The decisive confinement check: the range must be wholly within this sub-window.
                if off.checked_add(len).is_none_or(|end| end > size) {
                    return Ok(vec![EINVAL]);
                }
                Ok(vec![match op {
                    0 => mem.map(base + off, len, prot),
                    1 => mem.unmap(base + off, len),
                    2 => mem.protect(base + off, len, prot),
                    _ => EINVAL,
                }])
            }
            // The interpreter services `instantiate`/`join` in its eval loop (spawning a child vCPU
            // needs the executor the generic dispatch can't reach), so it never routes an Instantiator
            // here. A flat-window backend (the JIT) *does* — but only to **resolve this handle's
            // authority**: op 0 returns the carve range `[base, base+size)` so the JIT can compile and
            // run the child confined to a sub-window of it (the JIT owns the actual spawn). Other ops
            // are inert here (the JIT routes `join` to its own child table, never to the Host).
            Binding::Instantiator { base, size } => match op {
                0 => Ok(vec![base as i64, size as i64]),
                _ => Err(Trap::CapFault),
            },
            // The §14 `Yielder` (co-fiber `yield`) is serviced by the eval loop (it suspends the
            // running coroutine's continuation, which the generic dispatch can't); reaching here means
            // a `Yielder` cap.call slipped through (e.g. the JIT, which has no coroutine runtime) —
            // inert `CapFault`.
            Binding::Yielder => Err(Trap::CapFault),
            // A §14 `Module` handle confers instantiation authority only (through the Instantiator's
            // module ops); it has no callable ops — and crucially, the generic dispatch never exposes
            // the grant's host-side data, so no host pointer is guest-reachable.
            Binding::Module(_) => Err(Trap::CapFault),
            // §9/§12 IoRing. op 0 `submit(sq_ptr, n, cq_ptr)` — synchronous batch (increment 1/2). op 1
            // `submit_async(sq_ptr, n, counter_addr)` — kick the batch onto the pool and return; each
            // completion posts to the ring's [`RingState`] + bumps the in-window futex counter + wakes
            // a parked vCPU (increment 3). op 2 `reap(cq_ptr, max)` — flush ready completions to the
            // window on the vCPU thread.
            Binding::IoRing(idx) => match op {
                0 => self.io_ring_submit(args, mem),
                1 => self.io_ring_submit_async(idx, args, mem),
                2 => self.io_ring_reap(idx, args, mem),
                _ => Ok(vec![EINVAL]),
            },
            // §12 Blocking: `op 0 work(arg) -> mix(arg)`. As a *direct* cap.call it runs inline and
            // blocks the caller (the degenerate single path); a batched `submit` instead overlaps it
            // on the offload pool. Either way the result is the same deterministic transform.
            Binding::Blocking(idx) => match op {
                0 => {
                    let arg = *args.first().unwrap_or(&0);
                    Ok(vec![self.blockings[idx as usize].run(arg)])
                }
                _ => Ok(vec![EINVAL]),
            },
        }
    }

    /// §9/§12 the **submit/complete ring** (io_uring-shaped). `submit(sq_ptr, n, cq_ptr)` reads `n`
    /// 64-byte SQEs from `[sq_ptr, …)` (each a *deferred `cap.call`*) and writes a 32-byte CQE to
    /// `[cq_ptr, …)` per entry; returns the count completed. One boundary crossing for `n` ops (the
    /// §1a interface-amortization win).
    ///
    /// **Two execution classes (increment 2 — the bounded offload pool):**
    /// - **Inline** — ops that touch the window or `&mut Host` (Clock, Memory, Stream, …) run on the
    ///   submitting thread through the normal dispatch, in SQE order, exactly as increment 1.
    /// - **Offloaded** — `Blocking` ops (window-independent, `&mut Host`-free) are handed to the
    ///   bounded [`OffloadPool`] and run **concurrently** on `K` threads (waves of `K`), so the
    ///   guest's vCPU thread isn't multiplied by the blocking count (§12 "0 blocked vCPU threads").
    ///
    /// Window reads (SQE parse) and writes (CQE) stay on the submit thread; only the offloaded *op
    /// bodies* overlap, and each `Blocking` result is a deterministic pure transform — so the final
    /// window is **identical to running every op inline in order**, and both backends still agree (the
    /// §18 oracle). The submit blocks until the whole batch completes (fiber-parking is increment 3).
    ///
    /// SQE (64 B, little-endian): `u32 type_id | u32 op | i32 handle | u32 n_args | i64 args[4] |
    /// i64 user_data | i64 pad`. CQE (32 B): `i64 user_data | i64 result | i64 status (0=ok, else a
    /// TrapKind code) | i64 pad`. A nested `IoRing` op, or an op the dispatch can't service
    /// (Instantiator/Yielder/Module → `CapFault`), simply lands as a CQE with a non-zero `status` —
    /// never a host panic and never unbounded recursion.
    fn io_ring_submit(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const SQE: u64 = 64;
        const CQE: u64 = 32;
        const MAX_SQ_ARGS: usize = 4;
        // A ring with no window is inert (`-EFAULT`); otherwise borrow the window once and reborrow it
        // (`&mut *m`) for each SQE's read, inline dispatch, and CQE write.
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let sq_ptr = *args.first().unwrap_or(&0) as u64;
        let n = (*args.get(1).unwrap_or(&0)).max(0) as u64;
        let cq_ptr = *args.get(2).unwrap_or(&0) as u64;

        // One pending completion per SQE we managed to read; filled inline now, or by the pool below.
        // (An unreadable SQE writes its `-EFAULT` CQE immediately and is not tracked here.)
        struct Pending {
            at: u64,
            user_data: i64,
            result: i64,
            status: i64,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(n as usize);
        // Offloadable `Blocking` SQEs: `(index into `pending`, its state, its argument)`.
        let mut offload: Vec<(usize, Arc<AsyncState>, i64)> = Vec::new();

        for i in 0..n {
            let at = cq_ptr + i * CQE;
            // Read SQE i (a borrow-checked window read; out-of-window ⇒ -EFAULT completion).
            let raw = match m.read_bytes(sq_ptr + i * SQE, SQE) {
                Some(r) => r,
                None => {
                    Self::write_cqe(&mut *m, at, 0, 0, -EFAULT);
                    continue;
                }
            };
            let type_id = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            let op = u32::from_le_bytes(raw[4..8].try_into().unwrap());
            let handle = i32::from_le_bytes(raw[8..12].try_into().unwrap());
            let n_args =
                (u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize).min(MAX_SQ_ARGS);
            let mut opargs = [0i64; MAX_SQ_ARGS];
            for (a, slot) in opargs.iter_mut().enumerate().take(n_args) {
                *slot = i64::from_le_bytes(raw[16 + a * 8..24 + a * 8].try_into().unwrap());
            }
            let user_data = i64::from_le_bytes(raw[48..56].try_into().unwrap());

            if type_id == iface::IO_RING {
                // A ring submitting to a ring would recurse without bound — inert CapFault.
                pending.push(Pending {
                    at,
                    user_data,
                    result: 0,
                    status: trap_status(&Trap::CapFault),
                });
            } else if type_id == iface::BLOCKING && op == 0 {
                // Offloadable iff the handle actually resolves to a `Blocking` binding; a forged /
                // wrong-type handle is an inert CapFault (the I2 check), never queued.
                match self.resolve(handle, iface::BLOCKING) {
                    Ok(Binding::Blocking(idx)) => {
                        let slot = pending.len();
                        pending.push(Pending {
                            at,
                            user_data,
                            result: 0, // filled from the pool below
                            status: 0,
                        });
                        offload.push((slot, Arc::clone(&self.blockings[idx as usize]), opargs[0]));
                    }
                    _ => pending.push(Pending {
                        at,
                        user_data,
                        result: 0,
                        status: trap_status(&Trap::CapFault),
                    }),
                }
            } else {
                // Inline: window-/host-touching ops run on the submit thread, in order.
                let (result, status) = match self.cap_dispatch_slots(
                    type_id,
                    op,
                    handle,
                    &opargs[..n_args],
                    Some(&mut *m),
                ) {
                    Ok(res) => (res.first().copied().unwrap_or(0), 0),
                    Err(t) => (0, trap_status(&t)),
                };
                pending.push(Pending {
                    at,
                    user_data,
                    result,
                    status,
                });
            }
        }

        // Run the offloadable blocking ops concurrently on the bounded pool (created lazily so a Host
        // that never offloads spawns no threads). Each job writes its result by index; the submit
        // thread parks until the whole batch posts completion, then we copy results back in order.
        if !offload.is_empty() {
            let results: Arc<Vec<AtomicI64>> =
                Arc::new(offload.iter().map(|_| AtomicI64::new(0)).collect());
            let mut jobs: Vec<OffloadJob> = Vec::with_capacity(offload.len());
            for (k, (_slot, state, arg)) in offload.iter().enumerate() {
                let state = Arc::clone(state);
                let arg = *arg;
                let results = Arc::clone(&results);
                jobs.push(Box::new(move || {
                    results[k].store(state.run(arg), Ordering::SeqCst);
                }));
            }
            let pool = self
                .pool
                .get_or_insert_with(|| OffloadPool::new(OFFLOAD_POOL_THREADS));
            pool.run_batch(jobs);
            for (k, (slot, _, _)) in offload.iter().enumerate() {
                pending[*slot].result = results[k].load(Ordering::SeqCst);
            }
        }

        for p in &pending {
            Self::write_cqe(&mut *m, p.at, p.user_data, p.result, p.status);
        }
        Ok(vec![n as i64])
    }

    /// §9/§12 **async submit** (op 1, increment 3). `submit_async(sq_ptr, n, counter_addr)` reads `n`
    /// SQEs, kicks the **offloadable** (`Blocking`) ones onto the bounded pool, runs the inline ones
    /// immediately, and returns the count submitted **without waiting**. Each completion posts its CQE
    /// to the ring's host-side [`RingState`] and atomic-increments the 4-byte futex **completion
    /// counter** at `counter_addr`; an offloaded completion additionally `notify`s the counter key to
    /// wake a vCPU parked in `wait` on it — an I/O completion *is* a futex notify (DESIGN §12). The
    /// guest then parks on the counter, runs other fibers, and `reap`s once it advances.
    ///
    /// Requires the backend to expose the futex counter (`async_counter`) **and** the wake hook
    /// (`async_notify`); without them — the JIT pre-§3b, or the deterministic explorer — it returns
    /// `-EINVAL`, and the guest is expected to fall back to the synchronous `submit`. CQEs are written
    /// only by `reap` on the vCPU thread, so the single counter atomic is the *only* cross-thread
    /// window write an async ring performs.
    fn io_ring_submit_async(
        &mut self,
        ring_idx: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const SQE: u64 = 64;
        const MAX_SQ_ARGS: usize = 4;
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let sq_ptr = *args.first().unwrap_or(&0) as u64;
        let n = (*args.get(1).unwrap_or(&0)).max(0) as u64;
        let counter_addr = *args.get(2).unwrap_or(&0) as u64;

        // The backend must expose the futex counter handle + the wake hook, else there is no async
        // path here (the guest falls back to the synchronous `submit`).
        let counter = match m.async_counter(counter_addr) {
            Some(c) => c,
            None => return Ok(vec![EINVAL]),
        };
        let notify = match &self.async_notify {
            Some(f) => Arc::clone(f),
            None => return Ok(vec![EINVAL]),
        };
        let ring = Arc::clone(&self.rings[ring_idx as usize]);

        let mut jobs: Vec<OffloadJob> = Vec::new();
        let mut inline_done: u32 = 0; // completions ready before we return (counter bumped once below)
        for i in 0..n {
            let raw = match m.read_bytes(sq_ptr + i * SQE, SQE) {
                Some(r) => r,
                None => {
                    ring.completed.lock().unwrap().push_back((0, 0, -EFAULT));
                    inline_done += 1;
                    continue;
                }
            };
            let type_id = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            let op = u32::from_le_bytes(raw[4..8].try_into().unwrap());
            let handle = i32::from_le_bytes(raw[8..12].try_into().unwrap());
            let n_args =
                (u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize).min(MAX_SQ_ARGS);
            let mut opargs = [0i64; MAX_SQ_ARGS];
            for (a, slot) in opargs.iter_mut().enumerate().take(n_args) {
                *slot = i64::from_le_bytes(raw[16 + a * 8..24 + a * 8].try_into().unwrap());
            }
            let user_data = i64::from_le_bytes(raw[48..56].try_into().unwrap());

            if type_id == iface::BLOCKING && op == 0 {
                if let Ok(Binding::Blocking(bidx)) = self.resolve(handle, iface::BLOCKING) {
                    // Offload: compute on a pool thread, post the completion, then bump+notify so a
                    // parked vCPU wakes (the counter write happens-before the notify, so the futex
                    // compare-under-lock can't lose the wakeup).
                    let state = Arc::clone(&self.blockings[bidx as usize]);
                    let arg = opargs[0];
                    let ring = Arc::clone(&ring);
                    let counter = Arc::clone(&counter);
                    let notify = Arc::clone(&notify);
                    jobs.push(Box::new(move || {
                        let r = state.run(arg);
                        ring.completed.lock().unwrap().push_back((user_data, r, 0));
                        counter.increment(1);
                        notify(counter.key(), u32::MAX);
                    }));
                    continue;
                }
                // forged / wrong-type Blocking handle → inert CapFault completion (the I2 check).
                ring.completed.lock().unwrap().push_back((
                    user_data,
                    0,
                    trap_status(&Trap::CapFault),
                ));
                inline_done += 1;
            } else if type_id == iface::IO_RING {
                // A ring submitting to a ring would recurse without bound — inert CapFault.
                ring.completed.lock().unwrap().push_back((
                    user_data,
                    0,
                    trap_status(&Trap::CapFault),
                ));
                inline_done += 1;
            } else {
                // Inline: window-/host-touching ops run now on the submit thread.
                let (result, status) = match self.cap_dispatch_slots(
                    type_id,
                    op,
                    handle,
                    &opargs[..n_args],
                    Some(&mut *m),
                ) {
                    Ok(res) => (res.first().copied().unwrap_or(0), 0),
                    Err(t) => (0, trap_status(&t)),
                };
                ring.completed
                    .lock()
                    .unwrap()
                    .push_back((user_data, result, status));
                inline_done += 1;
            }
        }

        // Account the inline completions on the counter once (no wake — the guest can't be parked
        // during its own submit). Offloaded ones bump the counter as they finish.
        if inline_done > 0 {
            counter.increment(inline_done as u64);
        }
        if !jobs.is_empty() {
            let pool = self
                .pool
                .get_or_insert_with(|| OffloadPool::new(OFFLOAD_POOL_THREADS));
            pool.dispatch(jobs);
        }
        Ok(vec![n as i64])
    }

    /// §9/§12 **reap** (op 2). `reap(cq_ptr, max) -> n_reaped` pops up to `max` ready completions from
    /// the ring's [`RingState`] and writes them as 32-byte CQEs to `[cq_ptr, …)`, on the vCPU thread.
    fn io_ring_reap(
        &mut self,
        ring_idx: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const CQE: u64 = 32;
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let cq_ptr = *args.first().unwrap_or(&0) as u64;
        let max = (*args.get(1).unwrap_or(&0)).max(0) as u64;
        let ring = Arc::clone(&self.rings[ring_idx as usize]);
        let mut q = ring.completed.lock().unwrap();
        let mut i = 0u64;
        while i < max {
            let Some((ud, result, status)) = q.pop_front() else {
                break;
            };
            Self::write_cqe(&mut *m, cq_ptr + i * CQE, ud, result, status);
            i += 1;
        }
        Ok(vec![i as i64])
    }

    /// Write one 32-byte CQE (little-endian) at `at`. A bad address is dropped (the guest's bug; the
    /// `completed` count still reflects the SQEs the host ran).
    fn write_cqe(m: &mut dyn GuestMem, at: u64, user_data: i64, result: i64, status: i64) {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&user_data.to_le_bytes());
        b[8..16].copy_from_slice(&result.to_le_bytes());
        b[16..24].copy_from_slice(&status.to_le_bytes());
        let _ = m.write_bytes(at, &b);
    }

    /// `Stream` ops (§3e D43): 0 `read`, 1 `write`, 2 `close`. Buffers are `(ptr,len)`,
    /// borrow-only — the host reads/writes the guest window in place after the §7
    /// trampoline bounds-checks `[ptr,ptr+len) ⊆ [0,size)` (violation → `-EFAULT`).
    fn stream_op(
        &mut self,
        role: StreamRole,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let ret = |v: i64| Ok(vec![v]);
        match op {
            0 => {
                // read(buf, len) -> bytes read (>=0) or -errno; only stdin is readable.
                if role != StreamRole::In {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
                let n = (len as usize).min(avail.len());
                let chunk = avail[..n].to_vec();
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                if m.write_bytes(ptr, &chunk).is_none() {
                    return ret(EFAULT);
                }
                self.stdin_pos += n;
                ret(n as i64)
            }
            1 => {
                // write(buf, len) -> bytes written (>=0) or -errno; stdin is not writable.
                let sink = match role {
                    StreamRole::Out => &mut self.stdout,
                    StreamRole::Err => &mut self.stderr,
                    StreamRole::In => return ret(EINVAL),
                };
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                match m.read_bytes(ptr, len) {
                    Some(bytes) => {
                        sink.extend_from_slice(&bytes);
                        ret(len as i64)
                    }
                    None => ret(EFAULT),
                }
            }
            2 => ret(0), // close: no-op in the MVP (exit reclaims all)
            _ => ret(EINVAL),
        }
    }
}

// ----------------------------------------------------------------------------
// Linear memory — the confinement-masking *reference* (§4, invariant I1)
// ----------------------------------------------------------------------------

/// The **host** page size — the granularity of the protection model (RO/unmap) *and* the lazy
/// backing-store chunk. Queried so the interpreter's protection granularity matches the JIT's real
/// `mprotect` on the same host (§4 "pin page size", host-page default); both backends query the
/// same value, so they agree page-for-page on any platform (4 KiB / 16 KiB / …). Lazy paging keeps
/// interpreter memory bounded by what a (fuel-limited) run touches, so a huge declared window never
/// eagerly allocates — safe to fuzz.
fn host_page_size() -> u64 {
    match page_size::get() as u64 {
        0 => 4096,
        p => p,
    }
}

/// The granularity a `SharedRegion` map (§13) aligns to — distinct from [`host_page_size`] (the
/// protection granularity) because a *shared mapping* is coarser on Windows. On unix this is the
/// host page (`mmap(MAP_FIXED)` aliases at page granularity); on Windows it is the **allocation
/// granularity** (64 KiB), which `MapViewOfFile3` *requires* for both the placement address and the
/// section offset. Both the interpreter reference and the JIT's flat window report this for
/// `SharedRegion` op 3 (`region_page_size`), so a guest aligns its region maps to a single value that
/// works on every backend and the §13 differential stays in lockstep. `page_size::get_granularity`
/// returns `dwAllocationGranularity` on Windows and the page size on unix.
pub fn host_region_granularity() -> u64 {
    match page_size::get_granularity() as u64 {
        0 => host_page_size(),
        g => g,
    }
}

/// Explicit per-page state in the guest-visible address space (§3e Memory cap / §4).
///
/// A page absent from the map takes the **default for its region**: read+write inside the
/// initial backed prefix `[0, mapped)`, and *unmapped* in the reserved tail `[mapped, reserved)`
/// — so growth into the tail must be made explicit by a `map` (a [`PageProt::Rw`] entry). This is
/// what lets the guest `map`/`unmap`/`protect` sparsely across the whole reserved window (the §1a
/// "sparse address space / lazy page supply" capability), in lockstep with the JIT's real page
/// tables (an uncommitted page is `PROT_NONE` there and faults identically).
///
/// A committed *anonymous* page is zero-filled and lives in [`Mem::pages`]; a [`PageProt::Backed`]
/// page's bytes instead live in a §13 `SharedRegion` buffer (keyed in [`Mem::regions`]) — the
/// primitive behind aliasing / the magic-ring-buffer trick. Crucially the access path
/// ([`Mem::byte`]/[`Mem::set_byte`]) just redirects where a page's bytes live; loads/stores stay
/// ordinary masked accesses (zero overhead), exactly as §13 specifies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageProt {
    /// Explicitly `map`ped read-write — committed even in the reserved tail (where *absent* would
    /// mean unmapped). Within the initial prefix, plain read-write is left *absent* (the default),
    /// so this entry only appears for grown/re-committed pages.
    Rw,
    /// `protect`ed read-only: reads succeed, a store faults (the D40 const-segment mechanism).
    Ro,
    /// `unmap`ped: any access faults.
    Unmapped,
    /// §13 aliased page: its bytes live at `region_off` in the `SharedRegion` `region`
    /// ([`Mem::regions`]), not in an anonymous [`Mem::pages`] entry. `writable` mirrors the map
    /// `prot` (a store to a read-only alias faults). Two pages with the same `region` (mapped at
    /// different window offsets) name the same backing → aliasing.
    Backed {
        region: u32,
        region_off: u64,
        writable: bool,
    },
}

/// A guest linear-memory window. Confinement itself lives in [`svm_mask::Window`]
/// (the isolated, separately-fuzzed security unit, §4); `Mem` owns the lazily paged backing
/// store, threads accesses through that confinement, and carries the guest-visible page
/// protection map (`map`/`unmap`/`protect`, §3e). This is the semantics the JIT is
/// differential-tested against (§18).
struct Mem {
    window: Window,
    /// Host page size (`host_page_size()`): protection + storage-chunk granularity. Cached per
    /// `Mem` so every method shares the one host-queried value (matches the JIT's `mprotect`).
    page: u64,
    /// The anonymous-page backing: a [`svm_mem::Region`] (`#![forbid(unsafe_code)]`-friendly) sized
    /// to the window's reserved extent. On unix this is one demand-zeroed `mmap` — the shareable
    /// substrate parallel vCPUs run over with real hardware atomics (§12); elsewhere a paged
    /// fallback. §13 aliased pages live in `regions`, not here. Held in an `Arc` so a spawned vCPU
    /// (`thread.spawn`) shares the *same* bytes — `Region`'s accessors are all `&self`, so the `Arc`
    /// derefs transparently.
    back: Arc<Region>,
    /// The guest-visible **address space** — page-protection map + §13 region backings — behind a
    /// shared `RwLock` so all vCPUs of a run see one another's `map`/`unmap`/`protect` live (§12). A
    /// spawned vCPU ([`Mem::fork_for_thread`]) clones this `Arc`, sharing the same address space; the
    /// `RwLock` lets the many readers (every protection check) run concurrently while `map`/`unmap`
    /// take the brief write lock.
    space: Arc<RwLock<AddrSpace>>,
    /// Fast-path flag: set once any §13 region is aliased in (monotonic). While clear — the
    /// overwhelmingly common case — the per-byte path skips the address-space lock entirely and goes
    /// straight to `back`, since no page can be `Backed`. Shared with forked vCPUs.
    has_regions: Arc<AtomicBool>,
    /// §14 fault-driven-yield side-channel: the confined address of the most recent **recoverable**
    /// page fault (an in-window access to an unmapped/read-only page — `check_prot` sets it,
    /// `confine_checked` clears it to [`NO_FAULT`]). A coroutine child with `fault_yields` reads it
    /// after a `MemoryFault` to distinguish a recoverable page fault (suspend to the parent, which
    /// supplies the page) from an out-of-window fault (a real trap). Per-`Mem` (each vCPU owns its
    /// own), written/read only by the owning thread; `AtomicU64` keeps `Mem: Sync` for the futex path.
    last_fault: AtomicU64,
    /// Monotonic count of operations that **actually changed** a byte (a `store`/`atomic.store`/
    /// `atomic.rmw`, or an `atomic.cmpxchg` that *swapped*). The deterministic explorer reads the
    /// per-turn delta to drive spin-loop detection (a turn that changed no memory and returned the vCPU
    /// to the same local configuration is a pure spin → park it) and spin wakeups (a change wakes
    /// spinners parked on the written address). Per-`Mem` (only the running vCPU writes through its own).
    writes: u64,
}

/// Sentinel for [`Mem::last_fault`] meaning "no recoverable fault pending" — never a valid confined
/// address (every access is bounded to `< reserved ≤ 2^MAX_JIT_WINDOW_LOG2`).
const NO_FAULT: u64 = u64::MAX;

/// The shared, synchronized guest address space (§12): the page-protection map plus the §13 region
/// backings, mutated by `map`/`unmap`/`protect` and read by every access check. Lives behind
/// `Mem::space`'s `RwLock`; all vCPUs of a run share one.
#[derive(Default)]
struct AddrSpace {
    /// Page index (`offset / page`) ⇒ explicit page state. A page absent from the map takes its
    /// region default: read+write inside the initial prefix `[0, mapped)`, unmapped in the
    /// reserved tail `[mapped, reserved)`. Entries appear for `protect`ed (`Ro`), `unmap`ped
    /// (`Unmapped`), and grown/re-committed tail (`Rw`) pages — anywhere in `[0, reserved)`.
    prot: BTreeMap<u64, PageProt>,
    /// §13 `SharedRegion` backings this window has aliased in, keyed by region id (the bytes a
    /// [`PageProt::Backed`] page redirects to). A clone of the `Host`'s `Arc`, so two windows — or
    /// two offsets in one window — that map the same region share the *same* bytes.
    regions: BTreeMap<u32, RegionBacking>,
}

impl Mem {
    /// A window whose mask domain is `1 << reserved_log2` bytes but whose backed region is the
    /// declared `1 << mapped_log2` prefix; an access into the reserved-but-unmapped tail faults
    /// (the §4 "guard-when-bounded" model). `reserved_log2` is raised to at least `mapped_log2`,
    /// so passing `0` yields a fully-mapped window. Lazy paging means a huge mask domain (or
    /// reservation) never eagerly allocates.
    fn with_reservation(reserved_log2: u8, mapped_log2: u8) -> Mem {
        let reserved_log2 = reserved_log2.max(mapped_log2);
        let window = Window::with_mapped(reserved_log2, 1u64 << mapped_log2.min(63));
        let page = host_page_size();
        Mem {
            back: Arc::new(Region::new(window.reserved(), page)),
            window,
            page,
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// A fully-mapped **§14 sub-window**: a `1 << size_log2`-byte child window at absolute offset
    /// `base` inside a parent backing of `parent_bytes` bytes (the child runs over the parent's
    /// `Region`). The masking unit ([`svm_mask::Window::sub`], fuzzed as the escape hinge) confines
    /// every child access into `[base, base + size)`, so the child can reach **only its slice** — never
    /// the parent's other memory or outside the parent window. `base` is size-aligned by `Window::sub`;
    /// the whole slice is backed (no `map`-growth inside a child yet). The backing is sized to hold
    /// `[0, base + size)`.
    fn sub_window(base: u64, size_log2: u8, parent_bytes: u64) -> Mem {
        let window = Window::sub(base, size_log2, 1u64 << size_log2.min(63));
        let page = host_page_size();
        let need = window.base().saturating_add(window.reserved());
        Mem {
            back: Arc::new(Region::new(parent_bytes.max(need), page)),
            window,
            page,
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// Read/write the shared address space, recovering from a poisoned lock (the interpreter never
    /// panics while holding it) rather than propagating the panic.
    fn space_read(&self) -> std::sync::RwLockReadGuard<'_, AddrSpace> {
        self.space.read().unwrap_or_else(|e| e.into_inner())
    }
    fn space_write(&self) -> std::sync::RwLockWriteGuard<'_, AddrSpace> {
        self.space.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Build the memory view a spawned vCPU (`thread.spawn`) starts with (§12): it shares the **same**
    /// everything — the `Arc<Region>` bytes *and* the `Arc<RwLock<AddrSpace>>` address space — so a
    /// `map`/`unmap`/`protect` (or §13 alias) by any vCPU is immediately visible to the others.
    /// Confinement (`window`/`page`) is copied (identical for every vCPU of the run).
    fn fork_for_thread(&self) -> Mem {
        Mem {
            window: self.window,
            page: self.page,
            back: Arc::clone(&self.back),
            space: Arc::clone(&self.space),
            has_regions: Arc::clone(&self.has_regions),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// Build the memory view a **§14 nested child** runs over: it shares this (parent's) `Arc<Region>`
    /// bytes — so the parent intrinsically sees all of the child's bytes (the superset, §14) — but
    /// **confines** the child to the fully-mapped sub-window `[abs_base, abs_base + 2^size_log2)`
    /// (window-absolute, in the shared backing's coordinates). The child sees a zero-based `[0, size)`
    /// and cannot learn it is nested; masking ([`Window::sub`]) does the base+mask in one step.
    ///
    /// Unlike [`fork_for_thread`](Mem::fork_for_thread), the child gets its **own** address space (a
    /// fresh, empty page-protection map + §13 region set), *not* the parent's: page protections are a
    /// per-domain view, and the prot map is keyed window-relative, so a shared map would alias the
    /// child's pages onto the parent's (a child `unmap` of *its* page 0 would hit the parent's). The
    /// domains share **bytes**, not page-protection state — cross-domain memory sharing is §13, and
    /// lazy paging is the parent fielding the child's faults (co-fiber), not a shared map.
    fn nested_view(&self, abs_base: u64, size_log2: u8) -> Mem {
        Mem {
            window: Window::sub(abs_base, size_log2, 1u64 << size_log2.min(63)),
            page: self.page,
            back: Arc::clone(&self.back),
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed. A page
    /// absent from the map takes its region default — read+write in the initial prefix
    /// `[0, mapped)`, unmapped in the reserved tail (growth must be an explicit `map`).
    fn page_access(&self, prot: &BTreeMap<u64, PageProt>, page: u64) -> Option<bool> {
        match prot.get(&page) {
            Some(PageProt::Rw) => Some(true),
            Some(PageProt::Ro) => Some(false),
            Some(PageProt::Backed { writable, .. }) => Some(*writable),
            Some(PageProt::Unmapped) => None,
            None => (page * self.page < self.window.mapped()).then_some(true),
        }
    }

    /// Enforce the page state for a `width`-byte access at confined offset `base`: any access to an
    /// unmapped page, or a store to a read-only page, faults (§4/§5). Fast-pathed when the access
    /// lies wholly in the committed prefix and no page has been re-protected (the common case), so
    /// unprotected windows pay nothing.
    fn check_prot(&self, base: u64, width: u32, write: bool) -> Result<(), Trap> {
        // `base` is the *absolute* confined address; the page-map and `mapped` bound are
        // window-relative (`rel == base` for a top-level window; offset by the sub-window base for a
        // §14 child).
        let rel = base.wrapping_sub(self.window.base());
        let last = rel + width as u64 - 1;
        let space = self.space_read();
        if space.prot.is_empty() && last < self.window.mapped() {
            return Ok(());
        }
        for page in (rel / self.page)..=(last / self.page) {
            match self.page_access(&space.prot, page) {
                // A **recoverable** in-window page fault: record the confined address so a §14
                // coroutine child can suspend to its parent (fault-driven yield) instead of trapping.
                None => return Err(self.page_fault(base)), // unmapped
                Some(false) if write => return Err(self.page_fault(base)), // read-only store
                _ => {}
            }
        }
        Ok(())
    }

    /// Record `base` as the pending recoverable page fault (for §14 fault-driven yield) and return the
    /// `MemoryFault` to propagate. A normal guest treats it as a trap (detect-and-kill); a coroutine
    /// child reads the recorded address and suspends to its parent instead.
    fn page_fault(&self, base: u64) -> Trap {
        self.last_fault.store(base, Ordering::Relaxed);
        Trap::MemoryFault
    }

    /// Take the pending recoverable page-fault address (set by [`check_prot`], cleared by
    /// [`confine_checked`]), clearing it. `None` if the last `MemoryFault` was an out-of-window fault
    /// (a real trap), not a recoverable page fault.
    fn take_fault(&self) -> Option<u64> {
        match self.last_fault.swap(NO_FAULT, Ordering::Relaxed) {
            NO_FAULT => None,
            addr => Some(addr),
        }
    }

    /// Mark **every** page of this window unmapped — demand-paging it (§14 lazy paging): a coroutine
    /// child started this way faults on first access of each page, suspending to the parent, which
    /// supplies the page and resumes. The parent virtualizes the whole sub-window.
    fn demand_page(&self) {
        // `div_ceil` so a child smaller than one host page (e.g. a 4 KiB sub-window on a 16 KiB-page
        // host) still gets its single covering page marked — masking keeps its accesses in-window.
        let pages = self.window.reserved().div_ceil(self.page).max(1);
        let mut space = self.space_write();
        for p in 0..pages {
            space.prot.insert(p, PageProt::Unmapped);
        }
    }

    /// Supply the page containing the confined `abs_addr` (§14 lazy paging): mark it read-write
    /// **without zeroing**, so the bytes the parent placed in the shared backing survive — the
    /// faulting access then re-executes and reads them. Used by `resume` after a fault-driven yield.
    fn supply_page(&self, abs_addr: u64) {
        let page = abs_addr.wrapping_sub(self.window.base()) / self.page;
        self.space_write().prot.insert(page, PageProt::Rw);
    }

    /// Confine the final effective address into `[0, reserved)` (the masking security op, §4) and
    /// reject a `width`-byte access that would overrun the reserved domain. Per-page committed-ness
    /// is enforced separately by [`Mem::check_prot`] (the functional bound), so a masked-but-
    /// uncommitted page faults there — matching the JIT's `PROT_NONE` page tables.
    fn confine_checked(&self, addr: u64, offset: u64, width: u32) -> Result<u64, Trap> {
        // `confine` returns the **absolute** address `base + rel` (`base == 0` for a top-level window,
        // the sub-window base for a §14 child). The reserved-domain guard is on the window-relative
        // `rel`; per-page committed-ness is enforced by `check_prot` (also window-relative). The
        // returned absolute address indexes the (possibly parent-sized) backing.
        self.last_fault.store(NO_FAULT, Ordering::Relaxed); // fresh: clear any prior page fault
        let abs = self.window.confine(addr, offset);
        let rel = abs.wrapping_sub(self.window.base());
        match rel.checked_add(width as u64) {
            // An out-of-window fault is a real trap (not a recoverable page fault) — leave `last_fault`
            // cleared so `take_fault` returns `None` and the coroutine path propagates the trap.
            Some(end) if end <= self.window.reserved() => Ok(abs),
            _ => Err(Trap::MemoryFault),
        }
    }

    fn load(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Value, Trap> {
        let (_, rty, width, signed) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, false)?;
        let raw = self.read_le(base, width);
        Ok(decode_loaded(rty, width, signed, raw))
    }

    fn store(&mut self, addr: u64, offset: u64, op: StoreOp, v: Value) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, true)?;
        // `write_le` keeps only the low `width` bytes, so narrow stores truncate.
        self.write_le(base, width, store_bits(v));
        self.writes += 1;
        Ok(())
    }

    /// §12 atomics share the confinement + page-protection path with `load`/`store`, and add a
    /// **natural-alignment** requirement: a misaligned effective address traps (`MemoryFault`). The
    /// window base and mask domain are width-aligned, so checking the confined address suffices.
    /// Single-threaded, an atomic's *value* semantics equal the non-atomic op; the JIT lowers these
    /// to hardware atomics so they stay correct once threads exist (§12). All operate on the full
    /// `ty` width (`i32`/`i64`).
    fn check_align(&self, base: u64, width: u32) -> Result<(), Trap> {
        if base.is_multiple_of(width as u64) {
            Ok(())
        } else {
            Err(Trap::MemoryFault)
        }
    }

    /// Whether `base`'s page is a §13 aliased (`Backed`) page. A naturally-aligned ≤8-byte atomic
    /// lies wholly within one host page, so the single page of `base` decides. Aliased pages keep the
    /// value-correct `read_le`/`write_le` path (their bytes live in an `Rc` region, not `back`);
    /// anonymous pages get `back`'s real hardware atomics (§12).
    fn is_backed(&self, base: u64) -> bool {
        self.has_regions.load(Ordering::Relaxed)
            && matches!(
                self.space_read()
                    .prot
                    .get(&(base.wrapping_sub(self.window.base()) / self.page)),
                Some(PageProt::Backed { .. })
            )
    }

    /// §9/§12 async-ring completion counter: confine + validate a 4-byte futex counter address (same
    /// gate as an `i32` atomic), require a normal anonymous page (a §13 alias's atomics route through
    /// `read_le`, not `back`, so an offload worker couldn't reach it consistently), and hand back the
    /// `Arc<Region>` + confined key for a worker to atomic-increment (matching `atomic_value`'s
    /// non-backed path) before it `notify`s.
    fn async_counter_impl(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        let base = self.confine_checked(counter_addr, 0, 4).ok()?;
        if !base.is_multiple_of(4) || self.is_backed(base) {
            return None;
        }
        self.check_prot(base, 4, true).ok()?;
        Some(Arc::new(RegionCounter {
            region: Arc::clone(&self.back),
            off: base,
        }))
    }

    /// Validate a `<ty>.atomic.wait` address: confine it, require natural alignment, and require the
    /// page be readable (`map`/`unmap`/`protect` state) — the same gate as a same-width atomic load.
    /// Returns the confined base (the parking-lot key). (§12 futex)
    fn prepare_wait(&self, addr: u64, ty: IntTy) -> Result<u64, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, 0, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, false)?;
        Ok(base)
    }

    /// The current `width`-byte value at confined `base` (no checks; `prepare_wait` ran them). Used
    /// for the futex compare under the parking lock — real atomic for anonymous pages, value-correct
    /// for §13 aliases.
    fn atomic_value(&self, base: u64, width: u32) -> u64 {
        if self.is_backed(base) {
            self.read_le(base, width)
        } else {
            self.back.atomic_load(base, width)
        }
    }

    /// Confine an `atomic.notify` address to its parking-lot key. `notify` reads no memory, so only
    /// bounds confinement applies (no alignment or protection check). (§12 futex)
    fn confine_for_notify(&self, addr: u64) -> Result<u64, Trap> {
        self.confine_checked(addr, 0, 1)
    }

    fn atomic_load(&self, addr: u64, offset: u64, ty: IntTy) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, false)?;
        let raw = if self.is_backed(base) {
            self.read_le(base, width)
        } else {
            self.back.atomic_load(base, width)
        };
        Ok(atomic_decode(ty, raw))
    }

    fn atomic_store(&mut self, addr: u64, offset: u64, ty: IntTy, v: Value) -> Result<(), Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        if self.is_backed(base) {
            self.write_le(base, width, store_bits(v));
        } else {
            self.back.atomic_store(base, width, store_bits(v));
        }
        self.writes += 1;
        Ok(())
    }

    /// Read the old value, apply `op` with `v`, write the result back, return the **old** value.
    fn atomic_rmw(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        op: AtomicRmwOp,
        v: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let old = if self.is_backed(base) {
            let old = self.read_le(base, width);
            self.write_le(base, width, atomic_rmw_apply(ty, op, old, store_bits(v)));
            old
        } else {
            self.back.atomic_rmw(base, width, rmw_op(op), store_bits(v))
        };
        self.writes += 1;
        Ok(atomic_decode(ty, old))
    }

    /// If `*addr == expected`, write `replacement`; always return the **old** value.
    fn atomic_cmpxchg(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        expected: Value,
        replacement: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let want = store_bits(expected) & width_mask(width);
        let old = if self.is_backed(base) {
            let old = self.read_le(base, width); // already the low `width` bytes, zero-extended
            if old == want {
                self.write_le(base, width, store_bits(replacement));
            }
            old
        } else {
            self.back
                .atomic_cmpxchg(base, width, store_bits(expected), store_bits(replacement))
        };
        // Count a write only when the compare succeeded (a failed cmpxchg leaves memory unchanged —
        // the distinction the spin detector needs to tell a spinning retry from a real acquire).
        if old == want {
            self.writes += 1;
        }
        Ok(atomic_decode(ty, old))
    }

    /// Validate a `map`/`unmap`/`protect` range (§3e): the offset must be page-aligned and the
    /// whole `[offset, offset+len)` must lie within the **reserved** window `[0, reserved)` — the
    /// guest may now grow into the reserved tail `[mapped, reserved)`, not just the initial backed
    /// prefix. Returns the inclusive page-index range it covers, or `Err(EINVAL)`.
    fn prot_pages(&self, offset: u64, len: u64) -> Result<core::ops::RangeInclusive<u64>, i64> {
        // `offset` is **guest-relative** — the zero-based window the guest sees (the whole `GuestMem`
        // surface speaks guest coordinates; a §14 child names its own `[0, size)`, never its position
        // in the parent). The prot map is keyed by the same relative pages
        // (`check_prot`/`page_access`); only backing-store accesses add `window.base()`.
        if len == 0 || !offset.is_multiple_of(self.page) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.window.reserved() {
            return Err(EINVAL);
        }
        Ok((offset / self.page)..=((end - 1) / self.page)) // len need not be a page multiple; round up
    }

    /// Set one page's protection from cap `prot` bits: `WRITE` ⇒ read+write, `READ` only ⇒
    /// read-only, neither ⇒ unmapped. A read-write page in the initial prefix is left *absent*
    /// (its default); in the reserved tail it needs an explicit [`PageProt::Rw`] entry, since
    /// *absent* there means unmapped.
    /// Apply a `map`/`protect` protection to one page in the given prot map (the caller holds the
    /// address-space write lock). Uses `self`'s immutable `window`/`page` only.
    fn set_prot(&self, prot: &mut BTreeMap<u64, PageProt>, page: u64, flags: i32) {
        if flags & PROT_WRITE != 0 {
            if page * self.page < self.window.mapped() {
                prot.remove(&page); // read+write is the prefix default (no entry)
            } else {
                prot.insert(page, PageProt::Rw); // explicit commit in the reserved tail
            }
        } else if flags & PROT_READ != 0 {
            prot.insert(page, PageProt::Ro);
        } else {
            prot.insert(page, PageProt::Unmapped);
        }
    }

    /// Place initialized data segments at instantiation (§3a / D40): write every segment's bytes,
    /// then mark the pages of each `readonly` segment read-only (so the init writes themselves
    /// don't fault). RO protection is page-granular, so a producer keeps RO data on its own pages
    /// (the verifier already bounds each segment to `[0, size)`).
    fn init_data(&mut self, data: &[Data]) {
        self.init_data_at(data, 0);
    }

    /// Like [`init_data`], but place each (child-relative) segment at `win_base + offset` — the §14
    /// sub-window's slice base, so segment bytes and their read-only protections land in the child's
    /// region of the parent backing (matching the masking, which confines child accesses to
    /// `[win_base, win_base + size)`). `win_base == 0` is the ordinary top-level window.
    fn init_data_at(&mut self, data: &[Data], win_base: u64) {
        // Byte writes first (no §13 regions exist at init ⇒ `set_byte` is lock-free)...
        for d in data {
            for (i, &b) in d.bytes.iter().enumerate() {
                self.set_byte(win_base + d.offset + i as u64, b);
            }
        }
        // ...then the read-only protections, under one address-space write lock. The prot map is
        // keyed by window-relative page (the masking confines accesses to this window, and
        // `check_prot` looks up relative pages), so fold the window base out of the absolute address.
        let mut space = self.space_write();
        for d in data {
            if d.readonly && !d.bytes.is_empty() {
                let first = (win_base + d.offset).wrapping_sub(self.window.base());
                let last = first + d.bytes.len() as u64 - 1;
                for page in (first / self.page)..=(last / self.page) {
                    space.prot.insert(page, PageProt::Ro);
                }
            }
        }
    }

    /// Every page touched by `[ptr, ptr+len)` is committed (and writable, when `write`), and the
    /// range stays within `[0, reserved)`. The §7 borrow check: a buffer straddling an unmapped or
    /// (for writes) read-only page is rejected (`-EFAULT`), and grown tail pages are accepted.
    fn range_committed(&self, ptr: u64, len: u64, write: bool) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        if end > self.window.reserved() {
            return false;
        }
        if len == 0 {
            return true;
        }
        let space = self.space_read();
        (ptr / self.page..=(end - 1) / self.page)
            .all(|page| matches!(self.page_access(&space.prot, page), Some(w) if w || !write))
    }

    /// Borrow-validate and read a `(ptr, len)` capability buffer (§7): every page of
    /// `[ptr, ptr+len)` must be committed. Returns the bytes, or `None` (→ `-EFAULT`).
    /// Confinement holds regardless; this explicit check is the recoverable guest-bug
    /// path, not a safety boundary.
    fn read_bytes_impl(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.range_committed(ptr, len, false) {
            return None;
        }
        // `ptr` is guest-relative; `byte` indexes the (possibly parent-shared) backing absolutely.
        let base = self.window.base();
        Some((0..len).map(|k| self.byte(base + ptr + k)).collect())
    }

    /// Borrow-validate and write a `(ptr, len)` capability buffer (§7): every page must be
    /// committed and writable. `None` → `-EFAULT`.
    fn write_bytes_impl(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        if !self.range_committed(ptr, data.len() as u64, true) {
            return None;
        }
        let base = self.window.base();
        for (k, b) in data.iter().enumerate() {
            self.set_byte(base + ptr + k as u64, *b);
        }
        Some(())
    }

    fn read_le(&self, base: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            raw |= (self.byte(base + k) as u64) << (8 * k);
        }
        raw
    }

    fn write_le(&mut self, base: u64, width: u32, raw: u64) {
        for k in 0..width as u64 {
            self.set_byte(base + k, (raw >> (8 * k)) as u8);
        }
    }

    /// Read one byte; unwritten anonymous pages read as zero. A [`PageProt::Backed`] page redirects
    /// to its §13 region buffer (so an aliased page reads whatever the shared backing holds).
    fn byte(&self, off: u64) -> u8 {
        // Fast path: no §13 region is mapped, so no page can be `Backed` — go straight to `back`
        // without touching the address-space lock (the hot, overwhelmingly common case).
        if !self.has_regions.load(Ordering::Relaxed) {
            return self.back.byte(off);
        }
        let idx = (off % self.page) as usize;
        let space = self.space_read();
        // The prot map is keyed by window-relative page (base folds out; the within-page `idx` is
        // unchanged since the base is page-aligned).
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = space
            .prot
            .get(&(off.wrapping_sub(self.window.base()) / self.page))
        {
            return space
                .regions
                .get(region)
                .map_or(0, |r| r.read_byte(*region_off + idx as u64));
        }
        self.back.byte(off)
    }

    fn set_byte(&self, off: u64, b: u8) {
        if !self.has_regions.load(Ordering::Relaxed) {
            self.back.set_byte(off, b);
            return;
        }
        let idx = (off % self.page) as usize;
        let space = self.space_read();
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = space
            .prot
            .get(&(off.wrapping_sub(self.window.base()) / self.page))
        {
            // §13 aliased page: write through to the shared region backing.
            if let Some(r) = space.regions.get(region) {
                r.write_byte(*region_off + idx as u64, b);
            }
            return;
        }
        self.back.set_byte(off, b);
    }

    /// Seed the low bytes of the window from `init` (escape-oracle, §18). Bytes past the
    /// window size are ignored — confinement only concerns `[0, size)`.
    fn seed(&mut self, init: &[u8]) {
        let n = (init.len() as u64).min(self.window.mapped());
        for i in 0..n {
            self.set_byte(i, init[i as usize]);
        }
    }

    /// Snapshot the low `n` bytes of the window (clamped to the backed `mapped` extent).
    fn snapshot(&self, n: u64) -> Vec<u8> {
        let n = n.min(self.window.mapped());
        (0..n).map(|i| self.byte(i)).collect()
    }

    /// Seed the **whole parent backing** of a §14 sub-window (parent-absolute bytes), so the
    /// escape-oracle starts with non-zero bytes *outside* the child's slice — a child write that
    /// escaped its `[base, base+size)` slice would then perturb a byte the snapshot catches.
    fn seed_parent(&self, init: &[u8]) {
        for (i, &b) in init.iter().enumerate() {
            self.set_byte(i as u64, b);
        }
    }

    /// Snapshot the **whole parent backing** `[0, parent_bytes)` of a §14 sub-window (paired with
    /// the JIT's `compile_and_run_capture_sub`, which returns the full parent window).
    fn snapshot_parent(&self, parent_bytes: u64) -> Vec<u8> {
        (0..parent_bytes).map(|i| self.byte(i)).collect()
    }

    /// Snapshot the low `min(reserved, max(mapped, snap_cap))` bytes for the escape-oracle —
    /// **including grown reserved-tail pages** (a page absent from the map reads zero, matching the
    /// JIT's freshly-committed tail). Page-wise (one map lookup per committed page, not per byte) so
    /// widening past the backed prefix stays cheap.
    fn snapshot_window(&self, snap_cap: usize) -> Vec<u8> {
        let snap = self
            .window
            .reserved()
            .min(self.window.mapped().max(snap_cap as u64)) as usize;
        let mut out = vec![0u8; snap];
        self.back.read_into(0, &mut out); // anonymous bytes (untouched / grown-tail read as zero)
                                          // §13 aliased pages live in their region backing, not in `back` — fill them from there.
        let space = self.space_read();
        for (&idx, p) in &space.prot {
            let PageProt::Backed {
                region, region_off, ..
            } = p
            else {
                continue;
            };
            let start = (idx * self.page) as usize;
            if start >= snap {
                continue;
            }
            let n = (self.page as usize).min(snap - start);
            if let Some(r) = space.regions.get(region) {
                for k in 0..n {
                    out[start + k] = r.read_byte(*region_off + k as u64);
                }
            }
        }
        out
    }
}

impl GuestMem for Mem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.read_bytes_impl(ptr, len)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        self.write_bytes_impl(ptr, data)
    }

    /// §3e op 0 `map`: (re)commit pages with `prot`, zero-filling them (a fresh commit). Works
    /// anywhere in the reserved window `[0, reserved)` — including **growth** into the reserved
    /// tail `[mapped, reserved)`, the §1a sparse-address-space capability. Out-of-range /
    /// misaligned → `-EINVAL`.
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        {
            let mut space = self.space_write();
            for page in pages.clone() {
                self.set_prot(&mut space.prot, page, prot);
            }
        }
        for page in pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page); // commit ⇒ fresh zeroed page
        }
        0
    }

    /// §3e op 1 `unmap`: decommit pages — any later access faults, and a re-`map` reads zero.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        {
            let mut space = self.space_write();
            for page in pages.clone() {
                space.prot.insert(page, PageProt::Unmapped);
            }
        }
        for page in pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page);
        }
        0
    }

    /// §3e op 2 `protect`: change the protection of mapped pages without touching their backing
    /// (the D40 read-only const-segment mechanism: `protect(READ)` ⇒ later stores fault). A §13
    /// aliased page stays aliased — only its writability changes (or it `unmap`s if neither R nor W),
    /// so the shared bytes survive a `protect(READ)`.
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let mut space = self.space_write();
        for page in pages {
            if let Some(PageProt::Backed {
                region, region_off, ..
            }) = space.prot.get(&page).copied()
            {
                if prot & (PROT_READ | PROT_WRITE) == 0 {
                    space.prot.insert(page, PageProt::Unmapped);
                } else {
                    space.prot.insert(
                        page,
                        PageProt::Backed {
                            region,
                            region_off,
                            writable: prot & PROT_WRITE != 0,
                        },
                    );
                }
            } else {
                self.set_prot(&mut space.prot, page, prot);
            }
        }
        0
    }

    /// §13 op 0 `map`: alias `backing`'s `[region_off, region_off+len)` into the window at
    /// `[win_off, win_off+len)`. Both window offsets and the region offset round to whole pages; the
    /// region span must fit the backing; the mapping must be at least readable. The aliased pages'
    /// bytes then live in the region (a prior anonymous page there is dropped), so a store at one
    /// alias is visible at every other mapping of the same region.
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        region: u32,
        backing: RegionBacking,
    ) -> i64 {
        let pages: Vec<u64> = match self.prot_pages(win_off, len) {
            Ok(p) => p.collect(),
            Err(e) => return e,
        };
        if !region_off.is_multiple_of(self.page) || prot & PROT_READ == 0 {
            return EINVAL;
        }
        match region_off.checked_add(len) {
            Some(end) if end <= backing.size() => {}
            _ => return EINVAL,
        }
        let writable = prot & PROT_WRITE != 0;
        // A §13 alias now exists ⇒ the per-byte path must consult the address space from here on.
        self.has_regions.store(true, Ordering::Relaxed);
        {
            let mut space = self.space_write();
            space.regions.insert(region, backing);
            for (i, &page) in pages.iter().enumerate() {
                space.prot.insert(
                    page,
                    PageProt::Backed {
                        region,
                        region_off: region_off + i as u64 * self.page,
                        writable,
                    },
                );
            }
        }
        for &page in &pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page); // bytes live in the region now, not anonymous
        }
        0
    }

    /// §3e op 3 `page_size`: the backing-store page granularity (`self.page`, the host page) — the
    /// unit `map`/`unmap`/`protect` round to. The JIT's `MprotectWindow` reports the same host page,
    /// so the two backends agree.
    fn page_size(&self) -> i64 {
        self.page as i64
    }
    fn async_counter(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        self.async_counter_impl(counter_addr)
    }
}

/// Turn `width` raw little-endian bytes into the loaded value, sign- or zero-
/// extending narrow integer loads into the i32/i64 result type.
fn decode_loaded(rty: ValType, width: u32, signed: bool, raw: u64) -> Value {
    match rty {
        ValType::F32 => Value::F32(f32::from_bits(raw as u32)),
        ValType::F64 => Value::F64(f64::from_bits(raw)),
        ValType::I32 | ValType::I64 => {
            let bits = width * 8;
            let ext = if signed && bits < 64 {
                let shift = 64 - bits;
                (((raw << shift) as i64) >> shift) as u64 // arithmetic sign-extend
            } else {
                raw
            };
            if rty == ValType::I32 {
                Value::I32(ext as i32)
            } else {
                Value::I64(ext as i64)
            }
        }
    }
}

/// The low 64 bits of a value, for storing (the store width selects how many bytes).
fn store_bits(v: Value) -> u64 {
    match v {
        Value::I32(x) => x as u32 as u64,
        Value::I64(x) => x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
    }
}

/// Access width in bytes of an atomic `ty` (§12) — also its natural-alignment requirement.
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Map the IR's RMW op onto the memory substrate's (the substrate sits below `svm-ir`, so it carries
/// its own mirrored enum).
fn rmw_op(op: AtomicRmwOp) -> RmwOp {
    match op {
        AtomicRmwOp::Add => RmwOp::Add,
        AtomicRmwOp::Sub => RmwOp::Sub,
        AtomicRmwOp::And => RmwOp::And,
        AtomicRmwOp::Or => RmwOp::Or,
        AtomicRmwOp::Xor => RmwOp::Xor,
        AtomicRmwOp::Xchg => RmwOp::Xchg,
    }
}

/// Low-`width`-bytes mask (`width` ∈ {4, 8}).
fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Decode the low `ty`-width bytes (zero-extended, as from [`Mem::read_le`]) into a [`Value`].
fn atomic_decode(ty: IntTy, raw: u64) -> Value {
    match ty {
        IntTy::I32 => Value::I32(raw as i32),
        IntTy::I64 => Value::I64(raw as i64),
    }
}

/// Apply an atomic RMW: `old`/`arg` are the low `ty`-width bytes; returns the new low-`width` value.
fn atomic_rmw_apply(ty: IntTy, op: AtomicRmwOp, old: u64, arg: u64) -> u64 {
    match ty {
        IntTy::I32 => {
            let (o, a) = (old as u32, arg as u32);
            let r = match op {
                AtomicRmwOp::Add => o.wrapping_add(a),
                AtomicRmwOp::Sub => o.wrapping_sub(a),
                AtomicRmwOp::And => o & a,
                AtomicRmwOp::Or => o | a,
                AtomicRmwOp::Xor => o ^ a,
                AtomicRmwOp::Xchg => a,
            };
            r as u64
        }
        IntTy::I64 => match op {
            AtomicRmwOp::Add => old.wrapping_add(arg),
            AtomicRmwOp::Sub => old.wrapping_sub(arg),
            AtomicRmwOp::And => old & arg,
            AtomicRmwOp::Or => old | arg,
            AtomicRmwOp::Xor => old ^ arg,
            AtomicRmwOp::Xchg => arg,
        },
    }
}

fn bin32(op: BinOp, a: i32, b: i32) -> Result<i32, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i32::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u32) / (b as u32)) as i32
        }
        BinOp::RemS => {
            // `rem_s` traps only on a zero divisor. `INT_MIN % -1 == 0` — a perfectly
            // representable result, so it does *not* trap: traps are for results with no
            // representable value (§3b), and only the *quotient* overflows here, not the
            // remainder. (`wrapping_rem` yields 0.) See `div_s`, which does trap.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u32) % (b as u32)) as i32
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        // Shift amount is taken mod bitwidth (`wrapping_sh*` masks rhs to 0..31).
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u32).wrapping_shr(b as u32)) as i32,
        // Rotation amount is also mod bitwidth (`rotate_*` reduces it internally).
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

fn intun32(op: IntUnOp, a: i32) -> i32 {
    match op {
        IntUnOp::Clz => (a as u32).leading_zeros() as i32,
        IntUnOp::Ctz => (a as u32).trailing_zeros() as i32,
        IntUnOp::Popcnt => (a as u32).count_ones() as i32,
        IntUnOp::Extend8S => (a as i8) as i32,
        IntUnOp::Extend16S => (a as i16) as i32,
        IntUnOp::Extend32S => a, // identity for i32
    }
}

fn intun64(op: IntUnOp, a: i64) -> i64 {
    match op {
        IntUnOp::Clz => (a as u64).leading_zeros() as i64,
        IntUnOp::Ctz => (a as u64).trailing_zeros() as i64,
        IntUnOp::Popcnt => (a as u64).count_ones() as i64,
        IntUnOp::Extend8S => (a as i8) as i64,
        IntUnOp::Extend16S => (a as i16) as i64,
        IntUnOp::Extend32S => (a as i32) as i64,
    }
}

fn bin64(op: BinOp, a: i64, b: i64) -> Result<i64, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i64::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u64) / (b as u64)) as i64
        }
        BinOp::RemS => {
            // Only a zero divisor traps; `INT_MIN % -1 == 0` is representable (only the
            // quotient overflows, not the remainder), so it returns 0 — see `bin32`.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u64) % (b as u64)) as i64
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u64).wrapping_shr(b as u32)) as i64,
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

#[inline]
fn check_div(by_zero: bool, overflow: bool) -> Result<(), Trap> {
    if by_zero {
        Err(Trap::DivByZero)
    } else if overflow {
        Err(Trap::IntOverflow)
    } else {
        Ok(())
    }
}

fn cmp32(op: CmpOp, a: i32, b: i32) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u32) < (b as u32),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u32) <= (b as u32),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u32) > (b as u32),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u32) >= (b as u32),
    }
}

fn cmp64(op: CmpOp, a: i64, b: i64) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u64) < (b as u64),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u64) <= (b as u64),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u64) > (b as u64),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u64) >= (b as u64),
    }
}

#[inline]
fn step(fuel: &mut u64) -> Result<(), Trap> {
    *fuel = fuel.checked_sub(1).ok_or(Trap::OutOfFuel)?;
    Ok(())
}

#[inline]
fn get(vals: &[Value], v: ValIdx) -> Result<Value, Trap> {
    vals.get(v as usize).copied().ok_or(Trap::Malformed)
}

fn collect(vals: &[Value], idxs: &[ValIdx]) -> Result<Vec<Value>, Trap> {
    idxs.iter().map(|&v| get(vals, v)).collect()
}

#[inline]
/// Encode a value into its `i64` capability-ABI slot (scalars; `i32`/`f32` in the low
/// bits). Mirrors the JIT's marshalling so both drive the same slot-based dispatch.
fn val_to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}

/// Decode a capability-ABI result slot back to a `Value` of the declared type.
fn slot_to_val(ty: ValType, s: i64) -> Value {
    match ty {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
    }
}

fn as_i32(v: Value) -> Result<i32, Trap> {
    match v {
        Value::I32(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_i64(v: Value) -> Result<i64, Trap> {
    match v {
        Value::I64(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_f32(v: Value) -> Result<f32, Trap> {
    match v {
        Value::F32(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_f64(v: Value) -> Result<f64, Trap> {
    match v {
        Value::F64(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[cfg(test)]
mod prot_tests {
    //! White-box tests for the guest-visible page-protection model (`map`/`unmap`/`protect`,
    //! §3e Memory cap / §4) — the reference semantics the JIT's `mprotect`-backed side is
    //! differential-tested against next. Granularity is the **host** page size (4 KiB / 16 KiB),
    //! same as `Mem`, so these pass on any host.
    use super::*;

    /// The host page size — the protection granularity these tests align to.
    fn page() -> u64 {
        host_page_size()
    }

    /// A fully-mapped 64 KiB window (`mapped == reserved`, 16 pages).
    fn mem64k() -> Mem {
        Mem::with_reservation(0, 16)
    }

    #[test]
    fn protect_read_only_faults_store_allows_load() {
        let mut m = mem64k();
        let v = Value::I64(0x1122_3344_5566_7788u64 as i64);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        // a store to the RO page faults; the value is still readable
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        // an adjacent, unprotected page is unaffected
        assert!(m.store(page(), 0, StoreOp::I64, Value::I64(7)).is_ok());
    }

    #[test]
    fn protect_rw_restores_writability() {
        let mut m = mem64k();
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.protect(0, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unmap_faults_then_remap_zeroes() {
        let mut m = mem64k();
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x42)).is_ok());
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // re-commit ⇒ accessible again and zeroed
        assert_eq!(m.map(0, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    /// §12 shared synchronized address space: a forked vCPU view (`thread.spawn`) sees `map`/`unmap`
    /// made by another vCPU *after* the fork — the address space is shared, not snapshotted.
    #[test]
    fn forked_vcpu_sees_post_fork_mappings() {
        // 128 KiB reserved, 64 KiB mapped ⇒ the page at 64 KiB starts in the unmapped tail.
        let mut parent = Mem::with_reservation(17, 16);
        let child = parent.fork_for_thread();
        let tail = 1u64 << 16;
        // Both views fault on the tail initially (unmapped).
        assert_eq!(child.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        // Parent maps + writes the tail *after* the fork.
        assert_eq!(parent.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(parent
            .store(tail, 0, StoreOp::I64, Value::I64(0xCAFE))
            .is_ok());
        // The child now sees both the mapping (shared prot) and the bytes (shared region).
        assert_eq!(child.load(tail, 0, LoadOp::I64), Ok(Value::I64(0xCAFE)));
        // An unmap by the parent is likewise visible to the child.
        assert_eq!(parent.unmap(tail, page()), 0);
        assert_eq!(child.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
    }

    /// §14 nesting (interp `Mem` plumbing): a sub-window child confines every access into its own
    /// slice `[base, base + size)` of the parent backing — far/out-of-child offsets alias back into the
    /// slice, and every parent byte outside the slice is unreachable (stays zero). This is the
    /// interpreter half of running a guest in a nested child window.
    #[test]
    fn sub_window_child_confined_to_its_slice() {
        let base = 1u64 << 16; // child at 64 KiB
        let size_log2 = 12u8; // 4 KiB child
        let size = 1u64 << size_log2;
        let parent = 1u64 << 17; // 128 KiB parent backing
        let mut mem = Mem::sub_window(base, size_log2, parent);

        // A store at child offset 8 lands at absolute base+8; a far offset (size+8) wraps to slot 8.
        assert!(mem.store(8, 0, StoreOp::I64, Value::I64(0x1111)).is_ok());
        assert!(mem
            .store(size + 8, 0, StoreOp::I64, Value::I64(0x2222))
            .is_ok());
        assert_eq!(mem.load(8, 0, LoadOp::I64), Ok(Value::I64(0x2222))); // last write wins at slot 8
        assert_eq!(mem.confine_checked(8, 0, 8), Ok(base + 8)); // confined to the child's slice

        // Every (even wildly out-of-child) address aliases *into* `[base, base+size)` — never below
        // `base`, never at/above `base+size`.
        for &a in &[0u64, size, size * 1000, u64::MAX, base, parent] {
            let abs = mem.confine_checked(a, 0, 1).unwrap();
            assert!(
                abs >= base && abs < base + size,
                "child escaped its slice: {abs:#x}"
            );
        }

        // Decisive: every parent byte *outside* the child's slice is untouched (unreachable).
        for i in 0..parent {
            if i < base || i >= base + size {
                assert_eq!(
                    mem.back.byte(i),
                    0,
                    "child wrote outside its slice at {i:#x}"
                );
            }
        }
    }

    /// §14 nesting: a child's `AddressSpace`-style `map`/`unmap` (page protection) now works on a
    /// sub-window `Mem`. The prot map is keyed window-relative, so the base folds out consistently —
    /// before the fix, `unmap` on a sub-window `-EINVAL`'d (its absolute address was bounded against
    /// the child's window-relative `reserved`). A page unmapped via its **absolute** (§14-shifted)
    /// address faults a later plain access; re-`map` recommits it zeroed; and an address below the
    /// child's base or past its top is out of range.
    #[test]
    fn sub_window_page_protection_is_window_relative() {
        let base = 1u64 << 16; // child at 64 KiB
        let size_log2 = 16u8; // 64 KiB child (≥ one host page, so a whole page fits)
        let parent = 1u64 << 18; // 256 KiB parent backing
        let p = page();
        let mut mem = Mem::sub_window(base, size_log2, parent);

        // Initially fully mapped: a store/load at child offset 0 works.
        assert!(mem.store(0, 0, StoreOp::I64, Value::I64(0xABCD)).is_ok());
        assert_eq!(mem.load(0, 0, LoadOp::I64), Ok(Value::I64(0xABCD)));

        // Unmap the child's first page via its **guest-relative** offset 0 (the whole `GuestMem`
        // surface speaks the zero-based window the guest sees; the page lands at `base` in the
        // shared parent backing).
        assert_eq!(mem.unmap(0, p), 0, "sub-window unmap should succeed");
        assert_eq!(
            mem.load(0, 0, LoadOp::I64),
            Err(Trap::MemoryFault),
            "an access to the child's unmapped page must fault"
        );
        // A different page still within the child is unaffected.
        assert!(mem.store(p, 0, StoreOp::I64, Value::I64(0x1234)).is_ok());

        // Re-map recommits the page, zeroed — and the backing byte that changed is the *parent's*
        // byte at `base` (the child's slice), not the parent's page 0.
        assert_eq!(mem.map(0, p, PROT_WRITE), 0);
        assert_eq!(mem.load(0, 0, LoadOp::I64), Ok(Value::I64(0)));

        // The child cannot name anything at/past its own window top — its reserved domain is
        // `[0, size)`, wherever that sits in an ancestor's window.
        assert_eq!(
            mem.unmap(1u64 << size_log2, p),
            EINVAL,
            "at/after the child's window top is out of range"
        );
    }

    #[test]
    fn bad_args_einval() {
        let mut m = mem64k();
        assert_eq!(m.protect(1, page(), PROT_READ), EINVAL); // misaligned offset
        assert_eq!(m.protect(0, 0, PROT_READ), EINVAL); // zero length
                                                        // mem64k is fully mapped (reserved == mapped == 64 KiB), so its tail is empty: a range
                                                        // at/past the reserved top is still out of range.
        assert_eq!(m.unmap(65536, page()), EINVAL); // offset == reserved ⇒ out of range
        assert_eq!(m.map(0, 1 << 20, PROT_WRITE), EINVAL); // len past reserved
    }

    /// A window whose reserved mask domain (`1 MiB`) is larger than the initial backed prefix
    /// (`64 KiB`): the tail `[64 KiB, 1 MiB)` is reserved-but-unmapped and the guest can grow into
    /// it. `Mem::with_reservation(reserved_log2=20, mapped_log2=16)`.
    fn mem_growable() -> Mem {
        Mem::with_reservation(20, 16)
    }

    #[test]
    fn tail_access_faults_until_mapped() {
        let mut m = mem_growable();
        let tail = 1u64 << 16; // first byte of the reserved tail (64 KiB)
                               // Untouched tail faults (any access) — it is reserved-but-unmapped.
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // Grow one page into the tail; now it is committed, zeroed, read-write.
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(0x99)).is_ok());
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0x99)));
        // The next page up is still unmapped.
        assert_eq!(
            m.load(tail + page(), 0, LoadOp::I64),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn grow_then_unmap_faults_again() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(7)).is_ok());
        assert_eq!(m.unmap(tail, page()), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        // Re-mapping zero-fills (the old contents are gone).
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
    }

    #[test]
    fn grow_read_only_then_store_faults() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        // Map a tail page read-only: reads of the (zeroed) page succeed, a store faults.
        assert_eq!(m.map(tail, page(), PROT_READ), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn growth_bounds_are_reserved_not_mapped() {
        let mut m = mem_growable();
        let reserved = 1u64 << 20;
        // Mapping anywhere in the reserved tail is allowed now (was EINVAL pre-growth).
        assert_eq!(m.map(1 << 16, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.map(reserved - page(), page(), PROT_READ | PROT_WRITE), 0);
        // At/past the reserved top is still out of range.
        assert_eq!(m.map(reserved, page(), PROT_WRITE), EINVAL);
        assert_eq!(m.unmap(reserved - page(), 2 * page()), EINVAL);
    }

    #[test]
    fn grown_tail_buffer_borrow_round_trips() {
        // A cap buffer (§7 borrow) in a grown tail region validates and round-trips; one in the
        // unmapped tail is rejected (-EFAULT ⇒ None).
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_none()); // unmapped ⇒ EFAULT
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_some());
        assert_eq!(m.read_bytes_impl(tail, 4), Some(vec![1, 2, 3, 4]));
        // A borrow straddling the committed/uncommitted page boundary is rejected.
        assert!(m.read_bytes_impl(tail + page() - 2, 4).is_none());
    }

    #[test]
    fn cross_page_store_faults_if_either_page_protected() {
        let mut m = mem64k();
        // page 1 read-only; an 8-byte store straddling the page-0/1 boundary touches page 1.
        assert_eq!(m.protect(page(), page(), PROT_READ), 0);
        assert_eq!(
            m.store(page() - 4, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // fully within page 0 (still rw) is fine
        assert!(m.store(page() - 8, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unprotected_window_is_unrestricted() {
        // With an empty protection map, check_prot is a no-op: every in-window access works.
        let mut m = mem64k();
        for off in [0u64, 8, page(), 65536 - 8] {
            assert!(m.store(off, 0, StoreOp::I64, Value::I64(0x55)).is_ok());
            assert_eq!(m.load(off, 0, LoadOp::I64), Ok(Value::I64(0x55)));
        }
    }

    // ---- §13 SharedRegion: host-backed memory aliased into the window ----

    /// A §13 `SharedRegion` backing of `pages` whole host pages, zero-filled.
    fn region(pages: u64) -> RegionBacking {
        Arc::new(VecBacking(Mutex::new(vec![0u8; (pages * page()) as usize])))
    }

    #[test]
    fn shared_region_aliases_two_window_offsets() {
        // One region mapped at two window offsets names the *same* bytes: a store at one alias is
        // visible at the other (and vice versa) — the §13 zero-overhead aliasing primitive.
        let mut m = mem64k();
        let r = region(1);
        let (a, b) = (0, page());
        assert_eq!(
            m.map_region(a, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(b, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        let v = Value::I64(0x0123_4567_89ab_cdefu64 as i64);
        assert!(m.store(a, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.load(b, 0, LoadOp::I64), Ok(v), "A→B alias");
        let w = Value::I64(0x7777);
        assert!(m.store(b + 16, 0, StoreOp::I64, w).is_ok());
        assert_eq!(m.load(a + 16, 0, LoadOp::I64), Ok(w), "B→A alias");
    }

    #[test]
    fn shared_region_offsets_are_region_relative() {
        // Pointers are region-relative (§13): the same *region* offset at two window offsets aliases;
        // different region offsets are independent.
        let mut m = mem64k();
        let r = region(2);
        // window pages 0,1 ⇒ region pages 0,1.
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        // a second mapping of *region page 1* at window page 2.
        assert_eq!(
            m.map_region(2 * page(), page(), page(), PROT_READ | PROT_WRITE, 0, r),
            0
        );
        let v = Value::I64(0xdead_beef);
        assert!(m.store(page(), 0, StoreOp::I64, v).is_ok()); // write region page 1 via window page 1
        assert_eq!(m.load(2 * page(), 0, LoadOp::I64), Ok(v)); // observe via window page 2
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0))); // region page 0 independent
    }

    #[test]
    fn shared_region_read_only_alias_shares_reads_faults_stores() {
        let mut m = mem64k();
        let r = region(1);
        assert_eq!(
            m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(page(), 0, page(), PROT_READ, 0, r), 0); // RO alias of same region
        let v = Value::I64(0x5151_5151);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(
            m.load(page(), 0, LoadOp::I64),
            Ok(v),
            "RO alias sees the write"
        );
        assert_eq!(
            m.store(page(), 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault),
            "store to RO alias faults"
        );
        // protect(READ) on the RW alias keeps it aliased (shared bytes survive), now store-faulting.
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(2)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn shared_region_unmap_drops_alias_and_map_replaces_anonymous() {
        let mut m = mem64k();
        // Aliasing over an already-written anonymous page redirects to the region (old bytes gone).
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x4242)).is_ok());
        let r = region(1);
        assert_eq!(m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        assert_eq!(
            m.load(0, 0, LoadOp::I64),
            Ok(Value::I64(0)),
            "region zero-fill"
        );
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(9)).is_ok());
        // unmap drops the alias → faults.
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
    }

    #[test]
    fn shared_region_bad_args_einval() {
        let mut m = mem64k();
        let r = region(1); // one page
        assert_eq!(m.map_region(1, 0, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned window
        assert_eq!(m.map_region(0, 0, 0, PROT_READ, 0, r.clone()), EINVAL); // zero len
        assert_eq!(m.map_region(0, 1, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned region
        assert_eq!(
            m.map_region(0, page(), page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // region OOB
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // span > backing
        assert_eq!(m.map_region(0, 0, page(), PROT_WRITE, 0, r.clone()), EINVAL); // not readable
        assert_eq!(m.map_region(65536, 0, page(), PROT_READ, 0, r), EINVAL); // window past reserved
    }
}
