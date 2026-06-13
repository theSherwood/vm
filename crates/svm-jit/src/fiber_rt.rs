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
//! **Storage is domain-shared and fibers are MIGRATABLE (D57 steps 3b-ii + 3c).** The fiber table is
//! one [`SharedFiberTable`] per compiled module, shared by every vCPU of the domain — the same
//! unified handle namespace as the interpreter's run-shared registry (handles are `0, 1, …`
//! domain-wide; the §15 fiber quota is per-domain). Each slot carries the loom-verified single-owner
//! [`Ownership`] word (`fiber_registry`), and **any vCPU may resume any resumable fiber**: a
//! `cont.resume` *claims* the slot (`OWNED`-fresh or `RUNNABLE`-suspended → `RUNNING`,
//! [`Ownership::claim`] — exactly one racing claimant wins, a loser gets a clean `FiberFault`), a
//! voluntary suspend **publishes the fiber back to the pool** (`suspend_to_pool`, a release store),
//! and a return `finish`es the slot (`FREE`, generation bumped — a stale handle's claim fails). So a
//! fiber suspended on one OS thread continues on whichever thread claims it next — **stackful
//! migration**, matching the interpreter oracle's 3b-i semantics.
//!
//! **Why the cross-thread resume is sound (the 3c argument — SCHEDULING.md "Verification story"):**
//! the switch itself is the *same* `svm-fiber` instruction sequence that has always run — none of
//! the three ABIs touches thread-bound state (SysV/AAPCS64 save only callee-saved registers; the
//! MS-x64 switch swaps the TEB `StackBase`/`StackLimit`/`DeallocationStack` per switch, so "this
//! stack is active on this thread" is maintained wherever it runs). The only new requirement is a
//! happens-before edge from the suspending thread's last writes (the fiber's saved registers +
//! stack) to the claiming thread's first resume — exactly the `suspend_to_pool` release /
//! [`Ownership::claim`] acquire pairing, loom-verified in `fiber_registry`. Per-thread state the
//! fiber touches (`CURRENT_RT` for yielder pairing, the §5 guard recovery) is re-read **after**
//! every switch-in, never carried across a suspension; each vCPU thread arms its own guard, so a
//! fault in a migrated fiber unwinds the *resuming* thread's recovery (detect-and-kill, as ever).
//! This composition (verified protocol + real switch) cannot be model-checked — it is covered by
//! the **empirical net**: the randomized-migration interp↔JIT differential (`fiber_fuzz`), a
//! runtime single-owner assert at the resume seam ([`FiberSlot::running_on`]), guard-paged stacks,
//! and concurrent-steal stress (`jit_threads`).
//!
//! **Reentrancy/aliasing:** exactly one fiber of a chain is on a native stack at a time, and a fiber
//! whose handle is anywhere in a resume chain is `RUNNING`, so a re-entrant resume *loses the claim*
//! and faults (this replaces the old per-thread `chain` vec). No table lock or `&mut FiberRuntime`
//! is ever held across a switch — only a `*mut Fiber` to the boxed, address-stable fiber being
//! resumed, exclusive because its slot is `RUNNING` and only the claimant proceeds.

use crate::fiber_registry::Ownership;
use crate::{FnEntry, TrapKind};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use svm_fiber::{Fiber, State, Yielder};

thread_local! {
    /// The fiber runtime of the computation currently running on this OS thread — the standalone root,
    /// or the vCPU a scheduler is resuming. The `cont.*` thunks read it; each vCPU has its own
    /// *execution context* (yielders + owner token) while the fiber **table** is domain-shared
    /// ([`SharedFiberTable`]), so threads + fibers compose with one handle namespace. Null
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

/// Sentinel for [`FiberSlot::running_on`]: no vCPU is running this fiber.
const NOT_RUNNING: u64 = u64::MAX;

/// One slot of the domain-shared fiber table (D57 3b-ii/3c). The `Arc` keeps a resolved slot stable
/// while the table grows (a re-entrant `cont.new` from inside a running fiber pushes new slots).
pub(crate) struct FiberSlot {
    /// The loom-verified single-owner state word (`fiber_registry`): the migration arbiter — a
    /// `cont.resume` claims through it, from **any** vCPU (3c).
    own: Ownership,
    /// The **runtime single-owner assert** (empirical-net layer #3, SCHEDULING.md): the vCPU token
    /// currently running this fiber, [`NOT_RUNNING`] when parked. Set (and checked) right after a
    /// won claim, cleared right before the slot is republished — so if the claim protocol were
    /// ever mis-wired, a double-resume aborts loudly at the seam instead of silently running one
    /// native stack on two threads. Purely diagnostic: exclusivity itself is the `Ownership` CAS.
    running_on: AtomicU64,
    /// The parked native fiber. `Some` while the slot is `OWNED` (fresh) or `RUNNABLE`
    /// (suspended); the box stays in place during a resume (`RUNNING` guarantees the claimant
    /// exclusive access) and is dropped — its stack unmapped — when the fiber returns (`finish`).
    fiber: Mutex<Option<Box<Fiber>>>,
}

/// The **domain-shared fiber table** (D57 3b-ii): one per compiled module, shared by the root vCPU
/// and every `thread.spawn`ed vCPU — the unified handle namespace (slot index = the guest handle,
/// exactly the interpreter registry's numbering) and the per-domain §15 fiber quota. Slots are not
/// recycled yet (matching the interp registry; recycling + generation-carrying handles are a later
/// slice on both backends together — `finish` already bumps the slot generation under the hood).
pub(crate) struct SharedFiberTable {
    slots: Mutex<Vec<Arc<FiberSlot>>>,
    /// §15 quota: max fibers (incl. the implicit root computation) for the **whole domain**,
    /// clamped to [`MAX_FIBERS`] — per-run like the interpreter's, not per-vCPU.
    max_fibers: usize,
    /// Owner-token allocator: each vCPU's `FiberRuntime` takes a unique token at construction.
    next_owner: AtomicU64,
}

impl SharedFiberTable {
    pub(crate) fn new(max_fibers: usize) -> SharedFiberTable {
        SharedFiberTable {
            slots: Mutex::new(Vec::new()),
            max_fibers: max_fibers.clamp(1, MAX_FIBERS),
            next_owner: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Arc<FiberSlot>>> {
        self.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Quota pre-check (no allocation yet): would one more fiber exceed the domain budget? Checked
    /// *before* the fiber's stack is mmap'd so a fiber-bomb is a clean `FiberFault` that never
    /// touches the OS map limit.
    fn has_room(&self) -> bool {
        self.lock().len() + 1 < self.max_fibers
    }

    /// Allocate the next slot for a fresh (`OWNED`) fiber; the returned slot index is the guest
    /// handle. `None` if a racing allocation filled the domain quota since [`Self::has_room`].
    fn create(&self, fiber: Box<Fiber>) -> Option<i32> {
        let mut t = self.lock();
        if t.len() + 1 >= self.max_fibers {
            return None;
        }
        t.push(Arc::new(FiberSlot {
            own: Ownership::new_owned(),
            running_on: AtomicU64::new(NOT_RUNNING),
            fiber: Mutex::new(Some(fiber)),
        }));
        Some((t.len() - 1) as i32)
    }

    /// Resolve a (forgeable) handle: **masked** into the power-of-two-padded table (Spectre-safe,
    /// like `call_indirect` — and the same shape as the interp registry, so a forged handle now
    /// resolves over the same domain-wide namespace on both backends). Out of range ⇒ `None`.
    fn resolve(&self, handle: i32) -> Option<Arc<FiberSlot>> {
        let t = self.lock();
        let mask = t.len().next_power_of_two() - 1; // len 0 ⇒ mask 0 ⇒ slot 0, caught below
        let slot = (handle as u32 as usize) & mask;
        if slot >= t.len() {
            return None;
        }
        Some(Arc::clone(&t[slot]))
    }
}

/// Per-vCPU fiber execution context: the shared table plus this vCPU's identity and switch
/// bookkeeping. The table (the storage) is domain-shared; everything here is touched only by the
/// one OS thread running this vCPU.
pub(crate) struct FiberRuntime {
    /// The domain-shared fiber table (storage + ownership arbiter).
    table: Arc<SharedFiberTable>,
    /// This vCPU's owner token (the 3b-ii affinity identity).
    me: u64,
    /// The running fibers' `Yielder`s, one per live resume on this vCPU; `suspend` switches via
    /// the top one.
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
    pub(crate) fn new(
        table: Arc<SharedFiberTable>,
        fiber_type_id: u32,
        fn_table_mask: u64,
    ) -> FiberRuntime {
        let me = table.next_owner.fetch_add(1, Ordering::Relaxed);
        FiberRuntime {
            table,
            me,
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
// so no table lock or `&mut FiberRuntime` is ever held across a stack switch — borrows are taken only
// in short scopes that end *before* `resume`/`suspend`, and only a `*mut Fiber` (to an address-stable
// boxed fiber, kept alive by its slot `Arc`) crosses the switch. The `Ownership` claim makes each
// fiber's `&mut` exclusive — a fiber anywhere in a resume chain is `RUNNING`, so a re-entrant resume
// loses the claim — and slots are separate heap allocations from the table vec, so a re-entrant
// `cont.new` growing the table never moves a fiber being resumed.

/// `cont.new` thunk: allocate a suspended fiber that, on first resume, calls guest `funcref(sp, arg)`.
/// Returns the fiber handle (the domain-shared table's slot index — the same numbering as the interp
/// registry), or traps (`-1`) on a fiber-bomb (the **per-domain** §15 quota).
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
        let rt = &*rt;
        // Quota pre-check **before** the stack mmap, so a fiber-bomb is a clean `FiberFault` that
        // never exhausts the OS map limit. (`create` re-checks under the table lock — a racing
        // sibling vCPU may fill the last slot — at the cost of one transient stack allocation.)
        if !rt.table.has_room() {
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

    let fiber = Fiber::new(FIBER_STACK, move |y: &Yielder, arg: u64| -> u64 {
        // The *resuming* vCPU's runtime — read dynamically at **each** use, never carried across
        // a potential suspension: the body's start and its return may run on different OS threads
        // (3c migration), and the yielder push/pop must each target the thread actually running
        // the fiber at that moment (a push/pop pairs within one residency on one thread).
        // SAFETY: a fiber only runs under a resume, so `current()` is that thread's live runtime;
        // each `&mut` deref here is momentary and single-threaded.
        unsafe {
            (*current()).yielders.push(y as *const Yielder);
            // Resolve + type-check the funcref now (first resume), like the interpreter.
            let slot = (funcref as u32 as usize) & (mask as usize);
            let entry = (fn_table_base as *const FnEntry).add(slot);
            let result = if (*entry).type_id() != type_id {
                fault(trap_out);
                0u64
            } else {
                call_tramp((*entry).code(), mem_base, fn_table_base, trap_out, sp, arg)
            };
            (*current()).yielders.pop();
            result
        }
    });

    let rt = &*rt;
    match rt.table.create(Box::new(fiber)) {
        Some(handle) => handle,
        None => {
            fault(trap_out); // a sibling vCPU filled the domain quota since the pre-check
            -1
        }
    }
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
    // Phase 1: resolve + **claim** (D57): the slot must be this vCPU's (affinity, 3b-ii) and
    // `OWNED` — the `begin_owned` claim takes it to `RUNNING`, so a re-entrant resume (the fiber is
    // somewhere in a resume chain), a racing resume, or a finished fiber all *lose the claim* and
    // fault. No lock or `&mut` is held past this block; the `Arc` keeps the slot (and the boxed
    // fiber it owns) stable across the switch.
    let (slot, fib): (Arc<FiberSlot>, *mut Fiber) = {
        let rt = &*rt;
        let Some(slot) = rt.table.resolve(handle) else {
            fault(trap_out);
            *status_out = 1;
            return 0;
        };
        // **The 3c claim:** any vCPU may resume a fresh (`OWNED`) or suspended (`RUNNABLE`) fiber;
        // the acquire CAS arbitrates — exactly one racing claimant wins, and the winner
        // synchronizes-with the suspending thread's release, so the saved stack context is fully
        // visible even when *another* OS thread suspended it (the migration edge).
        if !slot.own.claim() {
            fault(trap_out);
            *status_out = 1;
            return 0;
        }
        // Runtime single-owner assert (empirical net #3): a won claim must find the seam clear.
        // `RUNNING` slots are unclaimable, so a non-sentinel here means the protocol wiring is
        // broken — abort loudly rather than run one native stack on two threads.
        let prev = slot.running_on.swap(rt.me, Ordering::Relaxed);
        assert!(
            prev == NOT_RUNNING,
            "single-owner violation: fiber claimed while running on vCPU {prev}"
        );
        let fib = match slot
            .fiber
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            Some(b) => &mut **b as *mut Fiber,
            // Unreachable by invariant (a claimable slot holds its fiber); fail closed, leaving
            // the slot claimed (inert thereafter) rather than aliasing anything.
            None => {
                fault(trap_out);
                *status_out = 1;
                return 0;
            }
        };
        (slot, fib)
    };
    // Phase 2: the switch (may reenter the runtime) — no lock or `&mut` held; the claim makes
    // `*fib` exclusive to this vCPU. The same `svm-fiber` instruction sequence regardless of which
    // thread the fiber last ran on (see the module header's 3c soundness argument).
    let st = (*fib).resume(arg as u64);
    // Phase 3: publish the fiber's new state (clearing the seam assert *before* republishing).
    match st {
        State::Yielded(v) => {
            // Voluntarily suspended: **publish to the pool** — claimable by any vCPU now, on any
            // thread (the migration point; release-pairs with the next claimant's acquire).
            slot.running_on.store(NOT_RUNNING, Ordering::Relaxed);
            slot.own.suspend_to_pool();
            *status_out = 0;
            v as i64
        }
        State::Complete(v) => {
            // Returned: drop the fiber (unmapping its stack) and free the slot — `finish` bumps
            // the generation, so any stale claim of this slot keeps failing.
            slot.fiber.lock().unwrap_or_else(|e| e.into_inner()).take();
            slot.running_on.store(NOT_RUNNING, Ordering::Relaxed);
            slot.own.finish();
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
    // Back from the suspension — possibly on a **different OS thread** (3c: another vCPU claimed
    // this fiber). Re-read `CURRENT_RT` rather than reusing the pre-switch `rt`: the yielder must
    // be pushed onto the *resuming* thread's runtime (each push/pop pairs within one residency on
    // one thread). Non-null by construction — a fiber only runs under a resume, which published it.
    {
        let rt = &mut *current();
        rt.yielders.push(y);
    }
    r as i64
}
