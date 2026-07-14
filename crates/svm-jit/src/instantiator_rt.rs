//! The JIT host side of the §14 **`Instantiator`** capability — VM-in-VM nesting. A guest holding an
//! `Instantiator` `instantiate`s a child confined to a power-of-two sub-window of its own window and
//! `join`s it. Unlike the interpreter (which spawns a child vCPU on its M:N executor), the JIT bakes
//! confinement into machine code, so a child confined to a *different* sub-window needs its own
//! compilation — "**nesting cost is paid at setup, not at runtime**" (§14): [`instantiate`] re-compiles
//! the child entry with the child's `mask`/`sub_base` ([`crate::compile_child_and_run`]) and runs it
//! over the **parent's live window** (so the parent intrinsically sees the child's writes — the §14
//! superset), under the caller's already-installed detect-and-kill guard.
//!
//! Authority lives in the host capability table (the same `Host` the interpreter uses): `instantiate`
//! resolves its `Instantiator` handle through the run's `cap.call` thunk (op 0 → the carve range
//! `[base, base+size)`), so a forged/wrong handle is an inert `CapFault` exactly as for any cap. The
//! child gets an **empty powerbox** for now (an inert `cap.call`); attenuated child caps + recursion +
//! "park only the calling fiber" (vs. today's synchronous run-at-`instantiate`) are follow-ups.

use crate::{mem, CapThunk, TrapKind};
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use svm_ir::{Data, Func, FuncIdx, ValType};

/// Negative-errno an out-of-range carve returns (matches the interpreter's `EINVAL`, §3e D42).
const EINVAL: i64 = -22;

/// Coroutine `resume` statuses, in lockstep with the interpreter's (`FIBER_SUSPENDED` /
/// `FIBER_RETURNED` / `CORO_FAULTED`): the child suspended at an explicit `yield`, returned, or
/// suspended on a demand-page fault (fault-driven yield; value = the fault address).
const SUSPENDED: i64 = 0;
const RETURNED: i64 = 1;
const FAULTED: i64 = 2;

/// Suspend-payload discriminants (the `u64` a fiber switch carries): an explicit `yield` (the value
/// is in [`CoroShared::yield_value`]) vs. a demand-page fault (the address in
/// [`CoroShared::fault_addr`], suspended from inside the fault handler).
const SUSP_YIELD: u64 = 0;
const SUSP_FAULT: u64 = 1;

/// The handle a coroutine child's `Yielder` capability is minted as — its single entry argument.
/// In lockstep with the reference `Host`'s **first grant** encoding (`(generation 1 << 8) | slot 0`,
/// see `svm_interp::Host::grant`): the interpreter's coroutine child gets its `Yielder` as the first
/// grant of a fresh powerbox, so both backends hand the child the *same* handle value (pinned by the
/// cross-backend differential — the handle is guest-visible data).
const YIELDER_HANDLE: i32 = 256;

/// Per-coroutine control stack (matches `fiber_rt::FIBER_STACK`): 256 KiB of reserved VA, guard-paged
/// by `svm-fiber`. On Windows the reservation is eager-committed (ISSUES.md I1), so this is real RAM
/// per live coroutine; a reservation the OS refuses surfaces as a trap, never an abort.
const CORO_STACK: usize = 1 << 18;

/// One spawned child's outcome: its `i64` result and trap cell (`0` = clean), plus whether it has
/// been `join`ed (a second join is inert — `ThreadFault`, matching the interpreter).
#[derive(Clone, Copy)]
struct Child {
    result: i64,
    trap: i64,
    joined: bool,
}

/// The per-run §14 nesting runtime, baked into the module's `Instantiator` `cap.call` sites. Holds
/// what compiling + running a child needs: the module's functions, the run's `cap.call` thunk/ctx
/// (to resolve an `Instantiator` handle's authority), and — supplied post-finalize via [`set_env`] —
/// the live window's detect-and-kill fault range. Children run synchronously at `instantiate` and
/// their outcomes are stashed for `join`.
pub(crate) struct Nursery {
    funcs: std::sync::Arc<[Func]>,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    /// §14 separate-module children: the host callback resolving a guest's `Module` handle to the
    /// granted module's code/data (`None` ⇒ module ops are an inert `CapFault`). Kept apart from the
    /// `cap.call` thunk so the host pointers it yields are never guest-reachable.
    resolve_module: Option<crate::ModuleResolver>,
    /// Address of the parent run's §5 kill-path interrupt cell (`0` ⇒ no kill-path armed). A nested
    /// JIT child is compiled to poll the **same** cell, so one host interrupt stops the parent *and*
    /// every child it spawned (a runaway child would otherwise hang the parent inside `instantiate` /
    /// `resume`, where the parent's own epoch checks can't fire).
    epoch_addr: usize,
    children: Mutex<Vec<Child>>,
    /// §14 co-fiber children (`spawn_coroutine`): suspended native continuations driven inline by
    /// `resume`, by handle (slot). `None` once finished (a later resume is an inert `CapFault`). The
    /// `Box` is taken out for the duration of a switch (so a re-entrant/concurrent resume of the same
    /// child sees `None` → `CapFault`), then reinserted if the child suspended.
    coros: Mutex<Vec<Option<Box<Coro>>>>,
    /// DURABILITY.md §4: the run is **durable** (set by [`Nursery::set_durable`] at run entry —
    /// the durable flag is applied after compile, where this nursery is built). A durable run's
    /// `instantiate`/`coro_spawn` **fail closed** (`-EINVAL`): this child runner re-compiles and
    /// runs children with no durable state (no shadow init, no instrumented-admission check), so a
    /// child it spawned could never drain-then-unwind — silently breaking "the snapshot unit is
    /// the domain closed over its nesting subtree". The interpreter is the reference for durable
    /// nesting; JIT parity is a follow-up.
    durable: AtomicBool,
    /// §4 freeze export: the §14 nested-child re-attach residue captured during a durable freeze — one
    /// [`crate::FrozenNested`] per child that unwound into its carve (`instantiate` records it when
    /// `compile_child_and_run` reports the child left `UNWINDING`). Drained by the top-level run after
    /// the freeze (`take_frozen_nested`). Depth-1 (root's direct child) for now; a grandchild's residue
    /// coalescing at the root (a shared sink) is a follow-up.
    frozen_nested_out: Mutex<Vec<crate::FrozenNested>>,
}

/// The slice of a coroutine's state its **child-side** thunks need (`coro_cap_thunk` — the `Yielder`),
/// boxed separately from [`Coro`] so its stable heap address can be baked into the child's compiled
/// code as the `cap.call` ctx *before* the [`Coro`] (which owns this box) is assembled. All access is
/// single-threaded (the child runs inline on the parent's thread), so `Cell`s suffice.
struct CoroShared {
    /// The running fiber's switch-back handle, set on fiber entry, null when the child isn't live on
    /// its stack — a `Yielder` cap.call outside that window is an inert `CapFault`.
    yielder: Cell<*const svm_fiber::Yielder>,
    /// The value the child handed to `yield`, read by the parent side after the switch.
    yield_value: Cell<i64>,
    /// The child-window VA of a pending demand fault ([`demand_cb`] sets it before suspending
    /// `SUSP_FAULT` from the fault handler; the parent side reads it after the switch).
    fault_addr: Cell<usize>,
    /// The child's trap cell (`0` = clean); a child trap propagates to the parent at `resume`.
    trap: UnsafeCell<i64>,
}

/// One §14 co-fiber child: its suspended native continuation (the fiber), its own guarded window,
/// its compiled code, and the guard-state snapshot swapped around every switch. Field order is drop
/// order: the fiber (stack) first, then the window, then the code (executable memory) — nothing on
/// an abandoned fiber stack holds a Rust destructor, so a mid-suspend teardown leaks nothing.
struct Coro {
    fiber: svm_fiber::Fiber,
    window: mem::GuestWindow,
    /// Keep-alive only: the fiber executes raw pointers into this compilation (extracted at spawn),
    /// so it must live exactly as long as the fiber can run.
    #[allow(dead_code)]
    code: crate::ChildCode,
    shared: Box<CoroShared>,
    /// The child's detect-and-kill recovery state while it is *suspended* (its guard is armed on its
    /// fiber stack); the parent swaps this with its own around every switch.
    guard: mem::GuardState,
    /// The carve's parent-window-absolute base + size — the slice synced with the child window at
    /// every switch boundary (the cooperative equivalent of the interpreter's live shared backing).
    sub_abs: u64,
    size: usize,
    /// Host page size — the demand-paging granularity and the sync-copy chunk.
    page: usize,
    /// Per-page committed flags. A demand-paged child (`spawn_demand_coroutine`) starts all-false
    /// (every first touch faults to the parent); a plain coroutine starts all-true. Sync copies (and
    /// nothing else) touch only committed pages — an uncommitted child page would fault the *host*.
    committed: Vec<bool>,
    /// Whether the child is demand-paged: its window range is registered as this thread's
    /// recoverable demand range around every switch into it.
    demand: bool,
    /// A demand fault awaiting supply: the child-window byte offset whose page the next `resume`
    /// commits (the parent has meanwhile written the bytes into its own slice) before re-running the
    /// rewound access.
    pending_fault: Option<usize>,
    /// The spawning thread: a suspended continuation's recovery state (sigjmp_buf / CONTEXT) is only
    /// valid on the thread that captured it, so a cross-thread `resume` is rejected (`CapFault`).
    thread: std::thread::ThreadId,
}

/// Sync the committed pages between the parent's window slice and the child's window —
/// `into_child` picks the direction. The cooperative-coroutine equivalent of the interpreter's live
/// shared backing: exact, because parent and child never run concurrently, so each side sees the
/// other's writes at every switch boundary. Uncommitted (never-supplied) pages are skipped — their
/// parent bytes are untouched by the child by construction (it faults instead of reaching them).
///
/// # Safety
/// `parent_base` is the live parent window; `[sub_abs, sub_abs+size)` is committed parent memory
/// and the child window's committed pages are mapped RW. Caller holds the only reference to `coro`.
unsafe fn sync_committed(coro: &Coro, parent_base: *mut u8, into_child: bool) {
    for (i, &committed) in coro.committed.iter().enumerate() {
        if !committed {
            continue;
        }
        let off = i * coro.page;
        let len = coro.page.min(coro.size - off);
        let parent = parent_base.add(coro.sub_abs as usize + off);
        let child = coro.window.base().add(off);
        if into_child {
            core::ptr::copy_nonoverlapping(parent, child, len);
        } else {
            core::ptr::copy_nonoverlapping(child, parent, len);
        }
    }
}

/// The demand-fault callback (§14 fault-driven yield), called by the fault handler — signal/VEH
/// context, on the **child's fiber stack** — for a fault in the child's registered demand range.
/// Records the address and suspends the child to its parent *from the handler frame* (the frame
/// stays live on the fiber stack across the suspension); when the parent has supplied the page and
/// resumes, `suspend` returns, we return nonzero, and the handler re-executes the faulting access.
/// With no live child (defensive; the range is only registered around a resume) returns 0 — the
/// fault falls through to detect-and-kill.
///
/// # Safety
/// `ctx` is the registered child's [`CoroShared`]; called only by the installed fault handler.
unsafe extern "C" fn demand_cb(addr: usize, ctx: *mut core::ffi::c_void) -> i32 {
    let sh = &*(ctx as *const CoroShared);
    let y = sh.yielder.get();
    if y.is_null() {
        return 0;
    }
    sh.fault_addr.set(addr);
    let _ = (*y).suspend(SUSP_FAULT);
    1
}

// SAFETY: the raw `cap_ctx` is the run's host pointer, valid for the whole run; the `Nursery` is
// only ever used on the run's threads while that host (and window) are alive. The interior tables
// are `Mutex`-guarded. (A child runs synchronously on the calling thread today, so there is in fact
// no cross-thread sharing yet; the bounds keep the door open for concurrent children later.)
unsafe impl Send for Nursery {}
unsafe impl Sync for Nursery {}

impl Nursery {
    pub(crate) fn new(
        funcs: std::sync::Arc<[Func]>,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        resolve_module: Option<crate::ModuleResolver>,
        epoch_addr: usize,
    ) -> Nursery {
        Nursery {
            funcs,
            cap_thunk,
            cap_ctx,
            resolve_module,
            epoch_addr,
            children: Mutex::new(Vec::new()),
            coros: Mutex::new(Vec::new()),
            durable: AtomicBool::new(false),
            frozen_nested_out: Mutex::new(Vec::new()),
        }
    }

    /// Drain the §14 nested-child freeze residue captured during a durable freeze (see
    /// [`Nursery::frozen_nested_out`]). Called by the top-level run after the root unwinds.
    pub(crate) fn take_frozen_nested(&self) -> Vec<crate::FrozenNested> {
        std::mem::take(
            &mut self
                .frozen_nested_out
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// Mark the run durable (DURABILITY.md §4) — see the [`Nursery::durable`] field: the nesting
    /// thunks then fail closed. Called at run entry (`run_code_raw`), after the entry wrappers
    /// have applied the compile-side durable flag.
    pub(crate) fn set_durable(&self, durable: bool) {
        self.durable.store(durable, Ordering::Release);
    }

    /// Resolve a spawn's child source (§14): `module < 0` ⇒ a **self** child (the parent's own
    /// functions, no data segments, no declared-memory constraint); otherwise a host-granted
    /// **`Module` handle** resolved via [`Nursery::resolve_module`] — the child runs *that* verified
    /// module's code, its data segments materialize into the carve, and the carve must equal its
    /// declared memory. `None` (with `*trap_out` set to a `CapFault`) for a forged handle or a run
    /// with no resolver.
    ///
    /// # Safety
    /// `trap_out` is the live trap cell. The returned slices borrow host-owned storage valid for the
    /// run (the [`ModuleResolver`](crate::ModuleResolver) contract).
    unsafe fn resolve_child(
        &self,
        module: i64,
        trap_out: *mut i64,
    ) -> Option<(&[Func], Option<i32>, &[Data])> {
        if module < 0 {
            return Some((&self.funcs, None, &[]));
        }
        let Some(resolver) = self.resolve_module else {
            *trap_out = TrapKind::CapFault as i64;
            return None;
        };
        let mut rm = core::mem::MaybeUninit::<crate::ResolvedModule>::zeroed().assume_init();
        if resolver(self.cap_ctx, module as i32, &mut rm) == 0 || rm.n_funcs == 0 {
            *trap_out = TrapKind::CapFault as i64;
            return None;
        }
        let funcs = std::slice::from_raw_parts(rm.funcs, rm.n_funcs);
        let data = if rm.n_data == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(rm.data, rm.n_data)
        };
        Some((funcs, Some(rm.memory_log2), data))
    }

    /// Resolve `handle` as this domain's `Instantiator` via the run's `cap.call` thunk, returning its
    /// carve range `[base, base+size)`. `None` (and `*trap_out` set) for a forged/closed/wrong handle.
    unsafe fn resolve(&self, mem_base: u64, handle: i32, trap_out: *mut i64) -> Option<(u64, u64)> {
        let mut out = [0i64; 2];
        // op 0 on an `Instantiator` binding returns `[base, size]` (see `cap_dispatch_slots`); a bad
        // handle sets `*trap_out` to a `CapFault` and we propagate by returning `None`.
        (self.cap_thunk)(
            self.cap_ctx,
            mem_base as *mut u8,
            0,
            0,
            svm_ir_iface_instantiator(),
            0,
            handle,
            core::ptr::null(),
            0,
            out.as_mut_ptr(),
            out.len() as u64,
            trap_out,
        );
        if unsafe { *trap_out } != 0 {
            return None;
        }
        Some((out[0] as u64, out[1] as u64))
    }
}

/// The `Instantiator` interface id (§3e), kept in lockstep with `svm_interp::iface::INSTANTIATOR`.
/// (`svm-jit` does not depend on `svm-interp`; the host dispatch on the other side checks the same
/// constant, and the cross-backend tests pin them equal.)
#[inline]
fn svm_ir_iface_instantiator() -> u32 {
    6
}

/// Materialize a §14 separate-module child's **data segments** into its carve `[abs_base, …+size)`
/// of the live parent window — exactly as if the child wrote them (the parent sees them, the §14
/// superset; the verifier bounded each segment to the child's declared window == the carve, with a
/// defensive re-check here). RO protection of `readonly` segments is skipped for nested children
/// (intra-domain self-corruption is a §1 non-goal).
///
/// # Safety
/// `[mem_base+abs_base, …+child_size)` is committed parent-window memory (the Instantiator bounded
/// the carve to the holder's range), valid for the call.
unsafe fn write_data_segments(data: &[Data], mem_base: u64, abs_base: u64, child_size: u64) {
    for d in data {
        if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
            core::ptr::copy_nonoverlapping(
                d.bytes.as_ptr(),
                (mem_base as *mut u8).add((abs_base + d.offset) as usize),
                d.bytes.len(),
            );
        }
    }
}

/// `instantiate(handle, [module,] entry, off, size_log2, fuel) -> child_handle` — the §14 nesting op
/// (`module < 0` ⇒ a self child, op 0; a `Module` handle ⇒ a **separate-module child**, op 5 — the
/// "plugin"). Resolves the holder's carve range, validates the requested power-of-two sub-window fits
/// within it (`-EINVAL` otherwise; a module child's carve must **equal its declared memory** — §14
/// transparency), materializes a module child's data segments into the carve, then **re-compiles**
/// the child entry confined to its own window and runs it (seeded from / copied back to the carve),
/// stashing its outcome for `join`. Returns a child handle (a table index), or `-EINVAL`. A child
/// that cannot be compiled (it uses §12 fibers/threads) or a forged module handle is a `CapFault`.
///
/// # Safety
/// Called from JIT'd code with `rt` the baked [`Nursery`], `mem_base` the live parent window base, and
/// `trap_out` the run's trap cell. All must be valid for the call (the JIT lowering guarantees it).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate(
    rt: *const Nursery,
    mem_base: u64,
    handle: i32,
    module: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    _fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    let durable = rt.durable.load(Ordering::Acquire);
    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    let Some((child_funcs, mod_mem, child_data)) = rt.resolve_child(module, trap_out) else {
        return 0; // forged Module handle / no resolver — CapFault set
    };
    // §4 (DURABILITY.md, "JIT parity" slice 1): a durable run may now nest a **same-module** child.
    // Its funcs are the parent's own instrumented funcs; a runnable same-module child on the JIT is a
    // pure-compute (non-may-suspend) func — it has no poll sites, so it runs atomically to completion
    // in its carve with no durable control-word setup needed (a would-be *instrumented* child hits a
    // `cap.call` against its empty powerbox → `CapFault`, so it never reaches an unwind). Freezing a
    // *live* nested child on the JIT — which needs the carve's ctx-0 control words + shadow base seeded
    // to match the interpreter — is the next slice. A durable **separate-module** child (`mod_mem =
    // Some`) stays fail-closed (host-supplied module identity + freeze residue are a later slice), as
    // does `coro_spawn`. Guest-reachable errno, like a bad carve.
    if durable && mod_mem.is_some() {
        return EINVAL as i32;
    }

    // The carve must be a power-of-two-aligned sub-window within `[0, size)` — a child can only get
    // what the holder sub-allocates (§14/D19) — and a module child's carve must equal its declared
    // memory. Bad entry index / size / alignment ⇒ `-EINVAL`.
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let mod_ok = mod_mem.is_none_or(|ml| ml == size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
        && (entry as usize) < child_funcs.len();
    if !fits || !mod_ok {
        return EINVAL as i32;
    }

    // A module child's data segments materialize into the carve now — `compile_child_and_run` seeds
    // the child's window from the carve, so they arrive exactly like the interpreter's shared-backing
    // writes at spawn.
    write_data_segments(child_data, mem_base, base + off, child_size);

    // The child entry takes its starter caps as `i64` args; with an empty powerbox today they are
    // unused, so pass zeros of the right arity (the entry is a fixed `(i64[, i64]) -> i64`).
    let nargs = child_funcs[entry as usize].params.len();
    let args = vec![0i64; nargs];

    // Re-compile the child as a top-level guest over its own window, seeded from the parent's
    // sub-region `[base+off, … + child_size)` and copied back on completion (the §14 superset).
    let outcome = crate::compile_child_and_run(
        child_funcs,
        entry as FuncIdx,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        &args,
        rt.epoch_addr, // §5: the child polls the parent's kill-path cell, so one interrupt kills both
        durable, // §4: seed the child's carve control words + give it an Instantiator powerbox
    );
    let (result, trap, unwound) = match outcome {
        Ok(rt) => rt,
        Err(_) => {
            // A child we cannot compile (fibers/threads, or a backend error) is a CapFault, not a
            // silent success — the guest learns its nesting request was refused.
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };

    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = children.len();
    children.push(Child {
        result,
        trap,
        joined: false,
    });
    // §4 freeze export: the child left its carve `UNWINDING` — it unwound mid-run under a freeze
    // instead of completing. Record its re-attach residue (depth-1: a direct child of the root, so
    // `parent_task = 0`). Its continuation lives in the carve (the frozen window image); this is what a
    // thaw needs to re-create the child domain. The `slot` is the child's join-table index — the
    // handle the guest holds — so a thaw resolves the reloaded handle to the re-attached child.
    if unwound {
        rt.frozen_nested_out
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(crate::FrozenNested {
                parent_task: 0,
                slot,
                carve_off: base + off,
                size_log2: size_log2 as u8,
                entry: entry as u32,
            });
    }
    slot as i32
}

/// `join(child_handle) -> result` — block on the child's completion (it already ran synchronously at
/// `instantiate` today) and return its `i64` result, propagating a child trap as the parent's
/// (`*trap_out`). A forged / already-joined handle is inert (a `CapFault`), matching the interpreter's
/// once-only join.
///
/// # Safety
/// As [`instantiate`]: `rt`/`trap_out` are the baked nursery + run trap cell, valid for the call.
pub(crate) unsafe extern "C" fn join(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i64 {
    let rt = &*rt;
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = handle as usize;
    match children.get_mut(slot) {
        Some(c) if !c.joined => {
            c.joined = true;
            if c.trap != 0 {
                *trap_out = c.trap; // a child trap propagates to the parent on join
                0
            } else {
                c.result
            }
        }
        _ => {
            *trap_out = TrapKind::CapFault as i64; // forged or already-joined handle
            0
        }
    }
}

/// The `Yielder` interface id (iface 7), in lockstep with `svm_interp::iface::YIELDER`.
#[inline]
fn svm_ir_iface_yielder() -> u32 {
    7
}

/// The §14 coroutine child's baked `cap.call` thunk: its powerbox holds exactly one capability — the
/// `Yielder` (iface 7, op 0 `yield(value) -> resumed`, handle [`YIELDER_HANDLE`]), which **suspends
/// the child's native stack** back to the parent's `resume`, handing over `value`; the next `resume`'s
/// value comes back as the cap.call's result. Anything else (wrong iface/op/handle, or a `Yielder`
/// call when the child is not live on its fiber) is an inert `CapFault`, matching the interpreter's
/// single-binding child powerbox.
///
/// # Safety
/// `ctx` is the child's baked [`CoroShared`]; `args`/`results` are the call-site slot buffers and
/// `trap_out` the child's trap cell — all valid for the call (the `cap.call` lowering guarantees it).
unsafe extern "C" fn coro_cap_thunk(
    ctx: *mut core::ffi::c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let sh = &*(ctx as *const CoroShared);
    if type_id == svm_ir_iface_yielder() && op == 0 && handle == YIELDER_HANDLE {
        let y = sh.yielder.get();
        if y.is_null() {
            *trap_out = TrapKind::CapFault as i64;
            return;
        }
        let value = if n_args >= 1 { *args } else { 0 };
        sh.yield_value.set(value);
        // Switch back to the parent's `resume` (which reads `yield_value`); when the parent resumes
        // us again, `suspend` returns the value it passed — the yield's result.
        let resumed = (*y).suspend(SUSP_YIELD);
        if n_results >= 1 {
            *results = resumed as i64;
        }
        *trap_out = 0;
    } else {
        *trap_out = TrapKind::CapFault as i64; // the child's powerbox holds only the Yielder
    }
}

/// A durable §14 child's baked `cap.call` thunk (DURABILITY.md §4, "JIT parity"): its powerbox holds
/// exactly one capability — an `Instantiator` over the child's **own full window** `[0, child_size)`,
/// so the child can carve and run a grandchild of its own. `Nursery::resolve` calls this with iface-6
/// op-0 to read the holder's `[base, size]`; the child is confined to its window by the masking
/// lowering and can forge no other cap, so any handle resolves to `[0, child_size]` (full authority
/// over its own window, and nothing beyond). Anything else is an inert `CapFault`, matching the
/// interpreter's single-binding child powerbox (`grant_instantiator(0, child_size)`).
///
/// # Safety
/// `ctx` points at a live `u64` (the child's window size) for the call; `results`/`trap_out` are the
/// call-site slot buffers (`Nursery::resolve` / the `cap.call` lowering guarantee them).
pub(crate) unsafe extern "C" fn child_instantiator_thunk(
    ctx: *mut core::ffi::c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    type_id: u32,
    op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    if type_id == svm_ir_iface_instantiator() && op == 0 && n_results >= 2 {
        let child_size = *(ctx as *const u64);
        *results = 0; // base, window-relative — the child's own window starts at 0
        *results.add(1) = child_size as i64; // size
        *trap_out = 0;
    } else {
        *trap_out = TrapKind::CapFault as i64;
    }
}

/// `spawn_coroutine(handle, entry, off, size_log2, fuel)` (Instantiator op 2; op 4 sets `demand`) —
/// compile the child confined to its own `2^size_log2` window and park it as a **suspended native
/// continuation** (a fiber that has not yet run), to be driven by [`coro_resume`]. Validation matches
/// the interpreter: the carve must be a power-of-two-aligned sub-window of the holder's range and the
/// entry a `(i64) -> (i64)` function (its argument is its `Yielder` handle) — else `-EINVAL`. A child
/// that cannot be compiled (it uses §12 fibers/threads) is a `CapFault`. Demand-paged spawn (op 4) is
/// not wired on the JIT yet — `CapFault`.
///
/// # Safety
/// As [`instantiate`]: `rt`/`mem_base`/`trap_out` are the baked nursery, live parent window base, and
/// run trap cell, all valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn coro_spawn(
    rt: *const Nursery,
    mem_base: u64,
    handle: i32,
    module: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    _fuel: i64,
    demand: i32,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    // §4 (DURABILITY.md): a durable run fails nesting closed — this runner can't run a durable
    // child (see the `Nursery::durable` field). Guest-reachable errno, like a bad carve.
    if rt.durable.load(Ordering::Acquire) {
        return EINVAL as i32;
    }
    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    let Some((child_funcs, mod_mem, child_data)) = rt.resolve_child(module, trap_out) else {
        return 0; // forged Module handle / no resolver — CapFault set
    };
    let demand = demand != 0;

    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    // A coroutine child entry is a fixed `(i64 yielder) -> (i64)`, matching the interpreter; a
    // module child's carve must equal its declared memory (§14 transparency).
    let ok_entry = child_funcs
        .get(entry as usize)
        .is_some_and(|f| f.params == [ValType::I64] && f.results == [ValType::I64]);
    let mod_ok = mod_mem.is_none_or(|ml| ml == size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size);
    if !ok_entry || !fits || !mod_ok {
        return EINVAL as i32;
    }

    // A module child's data segments materialize into the **carve** now: a plain coroutine's first
    // sync-in copies them into its window; a demand coroutine's pages stay unmapped, so its segments
    // are *supplied lazily*, page by page, as it first touches them (the §14 parent-as-pager model).
    write_data_segments(child_data, mem_base, base + off, child_size);

    // The child's own guarded window. Plain spawn: fully mapped, bytes synced with the parent's
    // slice at every switch (see `coro_resume`). Demand spawn (op 4): every page starts
    // *inaccessible* — the child's first touch of each page faults to the parent, which supplies it
    // (§14 fault-driven yield / lazy paging).
    let window = if demand {
        mem::GuestWindow::new_uncommitted(child_size as usize)
    } else {
        mem::GuestWindow::new(child_size as usize, child_size as usize)
    };
    let page = mem::page_size();
    let npages = (child_size as usize).div_ceil(page).max(1);
    let shared = Box::new(CoroShared {
        yielder: Cell::new(core::ptr::null()),
        yield_value: Cell::new(0),
        fault_addr: Cell::new(0),
        trap: UnsafeCell::new(0),
    });
    // Bake the child's code against its `Yielder` thunk + the stable `CoroShared` address.
    let code = match crate::compile_child(
        child_funcs,
        entry as FuncIdx,
        size_log2 as u8,
        coro_cap_thunk,
        &*shared as *const CoroShared as *mut core::ffi::c_void,
        rt.epoch_addr, // §5: the co-fiber child polls the parent's kill-path cell
        crate::InstEnv::null(), // a co-fiber child cannot itself nest (its Instantiator → CapFault)
    ) {
        Ok(c) => c,
        Err(_) => {
            *trap_out = TrapKind::CapFault as i64; // un-compilable child (fibers/threads/backend)
            return 0;
        }
    };

    // The suspended continuation: a fiber that, on first resume, runs the child's entry trampoline
    // under the child's own (re-entrantly nested) detect-and-kill guard. Captures only `Copy` raw
    // pointers — the `Coro` owns everything droppable, so a mid-suspend teardown leaks nothing.
    let sh_ptr: *const CoroShared = &*shared;
    let code_ptr = code.code;
    let fnt_ptr = code.fn_table.as_ptr() as *const core::ffi::c_void;
    let child_base = window.base();
    let (lo, hi) = window.fault_range();
    let Some(fiber) = svm_fiber::Fiber::new(CORO_STACK, move |y, _first| {
        // SAFETY: the fiber only runs inside `coro_resume` while the owning `Coro` (and so `sh_ptr`,
        // the code, and the window) is alive; all access is on the resuming thread.
        unsafe {
            (*sh_ptr).yielder.set(y as *const svm_fiber::Yielder);
            let args = [YIELDER_HANDLE as i64];
            let mut results = [0i64];
            let tc = (*sh_ptr).trap.get();
            let faulted = mem::run_guarded_range(
                code_ptr,
                args.as_ptr(),
                results.as_mut_ptr(),
                child_base,
                fnt_ptr,
                tc,
                lo,
                hi,
            );
            if faulted {
                *tc = mem::FAULT_TRAP;
            }
            (*sh_ptr).yielder.set(core::ptr::null());
            results[0] as u64
        }
    }) else {
        // The OS refused the coroutine control-stack reservation — recoverable, not an abort (I1).
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    };

    let coro = Box::new(Coro {
        fiber,
        window,
        code,
        shared,
        guard: mem::GuardState::new(), // disarmed until the child's first run arms it
        sub_abs: base + off,
        size: child_size as usize,
        page,
        committed: vec![!demand; npages], // demand: every page awaits its first-fault supply
        demand,
        pending_fault: None,
        thread: std::thread::current().id(),
    });
    let mut coros = rt.coros.lock().unwrap_or_else(|e| e.into_inner());
    coros.push(Some(coro));
    (coros.len() - 1) as i32
}

/// `resume(child, value) -> (status, value)` (Instantiator op 3) — drive the coroutine **inline** on
/// this thread until it `yield`s (SUSPENDED, the yielded value) or returns (RETURNED, its result; a
/// child trap propagates as the parent's). Around the switch the parent (a) syncs its window slice
/// with the child's window — the cooperative equivalent of the interpreter's live shared backing,
/// exact because the two never run concurrently — and (b) swaps the thread's detect-and-kill
/// recovery state, so the child's armed guard survives its suspension and the parent's is back in
/// force afterwards. A forged Instantiator handle, unknown/finished child handle, re-entrant resume
/// of a running child, or cross-thread resume is an inert `CapFault`.
///
/// # Safety
/// As [`instantiate`]; `status_out` is the call site's status slot, valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn coro_resume(
    rt: *const Nursery,
    mem_base: u64,
    handle: i32,
    child: i32,
    value: i64,
    status_out: *mut i64,
    trap_out: *mut i64,
) -> i64 {
    let rt = &*rt;
    *status_out = RETURNED;
    if rt.resolve(mem_base, handle, trap_out).is_none() {
        return 0; // forged Instantiator handle (`*trap_out` holds the CapFault)
    }
    // Take the child out of the table for the switch: a re-entrant resume of the same child (from
    // code the child itself reaches, or another vCPU) then sees `None` → CapFault, like the
    // interpreter's taken slot.
    let mut coro = {
        let mut coros = rt.coros.lock().unwrap_or_else(|e| e.into_inner());
        match coros.get_mut(child as usize).and_then(|c| c.take()) {
            Some(c) => c,
            None => {
                *trap_out = TrapKind::CapFault as i64;
                return 0;
            }
        }
    };
    if coro.thread != std::thread::current().id() {
        // A suspended continuation's recovery state is thread-affine (sigjmp_buf/CONTEXT); put the
        // child back for its rightful thread and fault this resume.
        rt.coros.lock().unwrap_or_else(|e| e.into_inner())[child as usize] = Some(coro);
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }

    // A pending demand fault: **supply the page** — commit it (fresh pages read zero), mark it
    // committed so the sync below copies the parent's bytes (placed there while the child sat
    // suspended) over it; the interpreter's supply-without-zeroing semantics, in two steps.
    if let Some(off) = coro.pending_fault.take() {
        let page_off = (off / coro.page) * coro.page;
        // SAFETY: `off < size` (the fault was inside the child's confined window), so the page lies
        // in the reservation.
        coro.window
            .commit_range(page_off, coro.page.min(coro.size - page_off).max(1));
        coro.committed[off / coro.page] = true;
    }

    // Sync in: the parent's slice → the child's window (committed pages only) — the parent may have
    // written bytes for the child since the last switch (e.g. supplying the faulted page).
    // SAFETY: see `sync_committed`; the parent slice was bounded by the Instantiator at spawn and
    // both windows live for the call.
    sync_committed(&coro, mem_base as *mut u8, true);

    // The switch, bracketed by (a) the demand-range registration, so a demand child's first touch of
    // an unsupplied page suspends back here instead of detect-and-kill, and (b) the guard-state
    // swap: install the child's recovery state (disarmed on first resume — its own guarded call arms
    // it), and capture it back (still armed if suspended) before reinstating the parent's.
    let mut parent_guard = mem::GuardState::new();
    mem::guard_save(&mut parent_guard);
    mem::guard_restore(&coro.guard);
    if coro.demand {
        let lo = coro.window.base() as usize;
        // SAFETY: the registration lives exactly for this switch (cleared below); the `CoroShared`
        // it points at is owned by `coro`, which we hold.
        mem::set_demand(
            lo,
            lo + coro.size,
            demand_cb,
            &*coro.shared as *const CoroShared as *mut core::ffi::c_void,
        );
    }
    let st = coro.fiber.resume(value as u64);
    if coro.demand {
        mem::clear_demand();
    }
    mem::guard_save(&mut coro.guard);
    mem::guard_restore(&parent_guard);

    // Sync out: the child's window → the parent's slice (committed pages only) — the parent sees
    // the child's writes (the §14 superset, materialized at every switch).
    // SAFETY: as the sync-in above.
    sync_committed(&coro, mem_base as *mut u8, false);

    match st {
        svm_fiber::State::Yielded(SUSP_FAULT) => {
            // Fault-driven yield: report `(FAULTED, the fault address in *parent-window*
            // coordinates)` and remember the page awaiting supply. The child sits suspended inside
            // its fault handler; the next resume commits the page, syncs the parent's bytes in, and
            // returns into the handler — which re-executes the faulting access.
            let child_off = coro.shared.fault_addr.get() - coro.window.base() as usize;
            let parent_addr = coro.sub_abs + child_off as u64;
            coro.pending_fault = Some(child_off);
            rt.coros.lock().unwrap_or_else(|e| e.into_inner())[child as usize] = Some(coro);
            *status_out = FAULTED;
            parent_addr as i64
        }
        svm_fiber::State::Yielded(_) => {
            let v = coro.shared.yield_value.get();
            rt.coros.lock().unwrap_or_else(|e| e.into_inner())[child as usize] = Some(coro);
            *status_out = SUSPENDED;
            v
        }
        svm_fiber::State::Complete(v) => {
            // Finished — the slot stays `None` (a later resume is an inert CapFault). A child trap
            // propagates to the parent, exactly like the synchronous child's `join`.
            let trap = *coro.shared.trap.get();
            if trap != 0 {
                *trap_out = trap;
                0
            } else {
                v as i64
            }
        }
    }
}
