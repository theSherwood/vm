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
//! **Why the cross-thread resume is sound (the 3c argument — DESIGN.md §23's verification story):**
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

use crate::fiber_registry::{Ownership, FIBER_HANDLE_GEN_MASK};
use crate::{FnEntry, TrapKind};
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
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

extern "C" {
    /// Publish the guest fiber handle now running on this thread into the shared `trap_capture.c`
    /// thread-local (per-fiber trap attribution, §5 W3 / §23-D57); returns the previous value so the
    /// resume seam can restore it. A no-op on the trap path until something traps — just a TLS store.
    fn svm_set_current_fiber(handle: i64) -> i64;
}

/// Max concurrently-allocated fibers per run (matches the interpreter's `MAX_FIBERS`): an anti-bomb
/// ceiling so a fiber-bomb traps (`FiberFault`) instead of exhausting host memory.
const MAX_FIBERS: usize = 1 << 16;

/// Per-fiber control-stack size (the out-of-band native stack, guard-paged by `svm-fiber`). 256 KiB:
/// large enough for deep guest call chains, but small enough that many concurrent fibers stay within
/// the host's address space — on Windows the reservation is **eager-committed**, so the per-fiber cost
/// is real RAM, not lazy VA (ISSUES.md I1). A reservation that the OS still refuses surfaces as a
/// `FiberFault`, never an abort, via the fallible `svm_fiber::Fiber::new`.
const FIBER_STACK: usize = 1 << 18;

// ---- Durable per-fiber shadow-stack layout (DURABILITY.md §12.8, D-fiber-cont option A) ----
//
// These MUST match `svm-interp`'s `SHADOW_SP_OFF` / `SHADOW_BASE` / `SHADOW_STRIDE` (the durable
// runtime ABI) — like `DURABLE_SNAPSHOT_PAGE`, svm-jit is TCB and can't depend on the interpreter,
// so the constants are duplicated; the cross-backend fiber freeze/thaw property catches drift. On a
// **durable** run the active shadow-SP word lives at `SHADOW_SP_OFF` in the window, and context `i`
// owns the shadow region `[SHADOW_BASE + i*SHADOW_STRIDE, +SHADOW_STRIDE)` — the root is context 0,
// a fiber in registry slot `s` is context `s+1`. The runtime keeps the active word pointing at the
// running context's region, swapping it on every fiber switch so a freeze spills into the right one.
const SHADOW_SP_OFF: u64 = 8;
const SHADOW_BASE: u64 = 64;
const SHADOW_STRIDE: u64 = 1 << 12;
/// Size of the reserved durable low slice (one 64 KiB wasm page); must match `svm-interp`'s
/// `DURABLE_RESERVE`. The per-context shadow regions live within `[0, DURABLE_RESERVE)`.
const DURABLE_RESERVE: u64 = 1 << 16;
/// Highest shadow-context index (must match `svm-interp`'s `MAX_SHADOW_CTX`): `DURABLE_RESERVE /
/// SHADOW_STRIDE - 1` = 15. Fibers grow **up** from context 1 (`slot+1`); spawned vCPUs grow **down**
/// from here (slice 3.3, mirroring the interp), so a `u16` mask holds every vCPU-context bit.
const MAX_SHADOW_CTX: usize = (DURABLE_RESERVE / SHADOW_STRIDE) as usize - 1;
/// Window byte offset of the durable state word (`NORMAL | UNWINDING | REWINDING`); the freeze
/// driver reads it to confirm a freeze is in progress. Must match `svm-interp`'s `STATE_OFF`.
const STATE_OFF: u64 = 0;
/// State-word value meaning "freeze in progress" (must match `svm-interp`'s `STATE_UNWINDING`).
const STATE_UNWINDING: i32 = 1;
/// State-word value meaning "thaw in progress" — a restored vCPU rewinds from its shadow extent
/// then flips to `NORMAL` and runs forward (must match `svm-interp`'s `STATE_REWINDING`).
const STATE_REWINDING: i32 = 2;
/// State-word value meaning "freeze armed" — the mid-run freeze trigger (must match
/// `svm-interp`'s `STATE_ARMED`). On an armed durable run the runtime counts down
/// [`ARM_COUNTDOWN_OFF`] at each fiber safepoint and promotes the word to `UNWINDING` at 0.
const STATE_ARMED: i32 = 3;
/// Window byte offset of the `i64` arm countdown (must match `svm-interp`'s `ARM_COUNTDOWN_OFF`).
const ARM_COUNTDOWN_OFF: u64 = 16;

/// Tick the **mid-run freeze trigger** at a fiber safepoint (`cont.resume`/`suspend`), the JIT mirror
/// of `svm_interp::Mem::durable_tick_arm`: if the run is `STATE_ARMED`, decrement the arm countdown
/// and, when it reaches 0, promote the state word to `UNWINDING` so the safepoint's trailing poll
/// begins the freeze. A no-op unless armed (one `i32` read in the common case), so an unarmed run is
/// byte-identical. Both backends count the same set — `cont.resume`/`suspend` (the ops routed through
/// runtime thunks) — so an armed freeze lands at the same safepoint on each (cross-backend parity).
///
/// # Safety
/// `mem_base` is a durable run's committed window base (`[STATE_OFF, ARM_COUNTDOWN_OFF + 8)` is RW
/// reserve). Only called on a durable run (the caller gates on `FiberRuntime::durable`).
unsafe fn window_tick_arm(mem_base: u64) {
    if *((mem_base + STATE_OFF) as *const i32) != STATE_ARMED {
        return;
    }
    let cd = (mem_base + ARM_COUNTDOWN_OFF) as *mut i64;
    let n = *cd - 1;
    *cd = n;
    if n <= 0 {
        *((mem_base + STATE_OFF) as *mut i32) = STATE_UNWINDING;
    }
}

/// The shadow-region base (window offset) of the fiber in registry `slot` (context `slot+1`).
fn fiber_region_base(slot: usize) -> u64 {
    SHADOW_BASE + (slot as u64 + 1) * SHADOW_STRIDE
}

/// Bits a fiber **guest handle** reserves for the registry slot; the rest carry a **generation**
/// (recycling step 1). MUST match `svm_interp`'s `FIBER_GEN_SHIFT` — the handle namespace is
/// cross-backend, so a frozen handle means the same on both. `MAX_FIBERS` bounds a slot to the low 16
/// bits; the `i64` handle leaves 48 bits for the generation. A handle is
/// `(generation << FIBER_GEN_SHIFT) | slot`; a fresh slot's generation is 0, so a non-recycled run's
/// handle is exactly its slot (byte-identical to before and to the interp).
const FIBER_GEN_SHIFT: u32 = 16;

/// Encode a fiber guest handle from its registry `slot` and (48-bit-masked) `generation`.
fn fiber_handle(slot: usize, generation: u64) -> i64 {
    (((generation & FIBER_HANDLE_GEN_MASK) << FIBER_GEN_SHIFT) | slot as u64) as i64
}

/// The generation a guest fiber handle carries (its high bits above the slot).
fn fiber_handle_generation(handle: i64) -> u64 {
    (handle as u64) >> FIBER_GEN_SHIFT
}

/// Read the active shadow-SP word from the durable window. `mem_base` is the window's host base.
/// # Safety: only called on a durable run, where `[SHADOW_SP_OFF, +8)` is committed RW reserve.
pub(crate) unsafe fn read_shadow_sp(mem_base: u64) -> u64 {
    *((mem_base + SHADOW_SP_OFF) as *const u64)
}

/// Write the active shadow-SP word into the durable window. # Safety: as [`read_shadow_sp`].
pub(crate) unsafe fn write_shadow_sp(mem_base: u64, sp: u64) {
    *((mem_base + SHADOW_SP_OFF) as *mut u64) = sp;
}

/// The shadow-region base (window offset) of a durable **vCPU** context `ctx` — context `i` owns
/// `[SHADOW_BASE + i*SHADOW_STRIDE, +SHADOW_STRIDE)` (must match `svm-interp`'s `shadow_region_base`).
/// Spawned vCPUs occupy the contexts the [`SharedFiberTable`] allocator hands out (top-down from
/// `MAX_SHADOW_CTX`); the inline single-worker path points the active shadow-SP word here before
/// running a child (slice 3.3).
pub(crate) fn shadow_region_base(ctx: usize) -> u64 {
    SHADOW_BASE + ctx as u64 * SHADOW_STRIDE
}

/// Whether the durable state word is **not** `NORMAL` (a freeze/thaw is in progress) — the gate for
/// running spawned children **inline** (single-worker, slice 3.3). `STATE_NORMAL` is 0 (matching
/// `svm-interp`). # Safety: `mem_base` is a durable run's committed window base.
pub(crate) unsafe fn window_is_durable_active(mem_base: u64) -> bool {
    *((mem_base + STATE_OFF) as *const i32) != 0
}

/// Set the durable state word to `REWINDING` — a thaw re-entry point (slice 3.3): a re-attached child
/// (and the root) starts in `REWINDING` to rewind from its restored shadow extent, then the
/// instrumented prologue flips the word to `NORMAL` and runs forward. # Safety: `mem_base` is a
/// durable run's committed window base.
pub(crate) unsafe fn window_set_rewinding(mem_base: u64) {
    *((mem_base + STATE_OFF) as *mut i32) = STATE_REWINDING;
}

/// The durable vCPU shadow **context** a restored shadow-SP lives in — the inverse of
/// [`shadow_region_base`] (`(sp − SHADOW_BASE) / SHADOW_STRIDE`, mirroring the interp's thaw). A thaw
/// derives a re-attached child's context from its frozen extent so the occupancy is rebuilt without a
/// separate record.
pub(crate) fn shadow_context_of_sp(sp: u64) -> usize {
    ((sp - SHADOW_BASE) / SHADOW_STRIDE) as usize
}

/// Whether the durable state word is `UNWINDING` (a freeze is in progress) — the entry path's gate
/// for running the [`freeze_drive`]. # Safety: `mem_base` is a durable run's committed window base.
pub(crate) unsafe fn window_is_unwinding(mem_base: u64) -> bool {
    *((mem_base + STATE_OFF) as *const i32) == STATE_UNWINDING
}

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
    /// The **runtime single-owner assert** (empirical-net layer #3, DESIGN.md §23): the vCPU token
    /// currently running this fiber, [`NOT_RUNNING`] when parked. Set (AcqRel swap) right after a
    /// won claim, cleared (Release store) right before the slot is republished — so if the claim
    /// protocol were ever mis-wired, a double-resume aborts loudly at the seam instead of silently
    /// running one native stack on two threads. Its Acquire/Release pair gives it a happens-before
    /// **independent of the `own` word** it cross-checks, so a mis-ordering of `own` cannot silence
    /// it in lockstep. Purely diagnostic: exclusivity itself is the `Ownership` CAS.
    running_on: AtomicU64,
    /// The parked native fiber. `Some` while the slot is `OWNED` (fresh) or `RUNNABLE`
    /// (suspended); the box stays in place during a resume (`RUNNING` guarantees the claimant
    /// exclusive access) and is dropped — its stack unmapped — when the fiber returns (`finish`).
    fiber: Mutex<Option<Box<Fiber>>>,
    /// **Durable** (DURABILITY.md §12.8): this fiber's saved shadow-SP — the extent of its
    /// continuation in its in-window shadow region. Initialized to the region base (empty); on a
    /// durable run the resume/return swap saves the live word here and restores it on the next
    /// resume. Touched only by the claimant (serialized by the `own` claim/publish), so `Relaxed`
    /// suffices; unused on a non-durable run.
    shadow_sp: AtomicU64,
    /// **Durable**: the fiber's entry funcref and data-stack base (the `cont.new` operands),
    /// retained so the freeze driver can export them in this fiber's [`crate::FrozenFiber`] residue
    /// (a thaw re-creates the fiber from them). Immutable after creation.
    func: i32,
    sp: i64,
}

/// The **domain-shared fiber table** (D57 3b-ii): one per compiled module, shared by the root vCPU
/// and every `thread.spawn`ed vCPU — the unified handle namespace (slot index = the guest handle,
/// exactly the interpreter registry's numbering) and the per-domain §15 fiber quota. Slots are not
/// recycled yet (matching the interp registry; recycling + generation-carrying handles are a later
/// slice on both backends together — `finish` already bumps the slot generation under the hood).
/// The fiber table's locked state: the slots, plus the freed-slot **min-heap** (recycling step 3).
struct TableState {
    slots: Vec<Arc<FiberSlot>>,
    /// Freed slots reclaimable for a new fiber, lowest first — the same policy as the interp registry
    /// (so handle values match across backends). A freed slot keeps its bumped generation, so reuse
    /// is ABA-safe; the table is bounded by the *peak concurrent* fiber count, not the lifetime total.
    free: BinaryHeap<Reverse<usize>>,
    /// **Occupied** durable vCPU shadow contexts (slice 3.3), a bitmask over contexts
    /// `1..=MAX_SHADOW_CTX` (bit `c` set ⇒ context `c` is live) — the JIT mirror of the interp
    /// registry's `vcpu_mask`. Spawned vCPUs grow **down** from `MAX_SHADOW_CTX` while fibers grow
    /// **up** from context 1; a child's bit is freed when it finishes, so the bound is *peak
    /// concurrent* vCPUs. Only touched on a durable run (state ≠ NORMAL ⇒ single-worker).
    vcpu_mask: u16,
}

pub(crate) struct SharedFiberTable {
    state: Mutex<TableState>,
    /// §15 quota: max fibers (incl. the implicit root computation) for the **whole domain**,
    /// clamped to [`MAX_FIBERS`] — per-run like the interpreter's, not per-vCPU.
    max_fibers: usize,
    /// Owner-token allocator: each vCPU's `FiberRuntime` takes a unique token at construction.
    next_owner: AtomicU64,
}

impl SharedFiberTable {
    pub(crate) fn new(max_fibers: usize) -> SharedFiberTable {
        SharedFiberTable {
            state: Mutex::new(TableState {
                slots: Vec::new(),
                free: BinaryHeap::new(),
                vcpu_mask: 0,
            }),
            max_fibers: max_fibers.clamp(1, MAX_FIBERS),
            next_owner: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, TableState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reserve the **highest free** durable vCPU shadow context above the fiber pool (slice 3.3,
    /// mirroring the interp's `reserve_vcpu_context`): spawned vCPUs grow down from `MAX_SHADOW_CTX`
    /// while fibers occupy contexts `1..=slots.len()`, so the picked context must stay clear of them.
    /// Reusing a freed (cleared) bit is the recycling that bounds the pool to peak-concurrent vCPUs.
    /// `None` if the reserve is full (the vCPU pool growing down would meet the fibers growing up).
    pub(crate) fn reserve_vcpu_context(&self) -> Option<usize> {
        let mut t = self.lock();
        let floor = t.slots.len(); // fibers occupy contexts 1..=slots.len()
        let mut c = MAX_SHADOW_CTX;
        while c > floor {
            if t.vcpu_mask & (1 << c) == 0 {
                t.vcpu_mask |= 1 << c;
                return Some(c);
            }
            c -= 1;
        }
        None
    }

    /// Free a spawned vCPU's shadow context for reuse (slice 3.3): called when the child genuinely
    /// finishes (a freeze-unwound child keeps it for thaw). A no-op for an out-of-range context.
    pub(crate) fn free_vcpu_context(&self, ctx: usize) {
        if (1..=MAX_SHADOW_CTX).contains(&ctx) {
            self.lock().vcpu_mask &= !(1 << ctx);
        }
    }

    /// Seed the durable vCPU-context occupancy a **thaw** re-establishes (slice 3.3): the re-attached
    /// children reclaim exactly the contexts they held at freeze (derived from their restored
    /// shadow-SPs), so a post-thaw spawn allocates into a genuinely-free context.
    #[allow(dead_code)] // wired by the inline single-worker path (slice 3.3 freeze side)
    pub(crate) fn seed_vcpu_mask(&self, mask: u16) {
        self.lock().vcpu_mask = mask;
    }

    /// Quota pre-check (no allocation yet): would one more fiber exceed the domain budget? Checked
    /// *before* the fiber's stack is mmap'd so a fiber-bomb is a clean `FiberFault` that never
    /// touches the OS map limit. A free slot is always room (recycling reuses, doesn't grow).
    fn has_room(&self) -> bool {
        let t = self.lock();
        !t.free.is_empty() || t.slots.len() + 1 < self.max_fibers
    }

    /// Allocate a slot for a fresh (`OWNED`) fiber; the returned guest handle carries the slot's
    /// generation. **Recycling (step 3):** the lowest freed slot is reused — its `Ownership` replaced
    /// at the kept (bumped) generation, so a stale handle to its former occupant still fails
    /// `claim_gen` — and only when none is free does the table grow. `None` if a racing allocation
    /// filled the domain quota since [`Self::has_room`].
    fn create(&self, fiber: Box<Fiber>, func: i32, sp: i64) -> Option<i64> {
        let mut t = self.lock();
        let reuse = t.free.peek().map(|&Reverse(s)| s);
        if reuse.is_none() && t.slots.len() + 1 >= self.max_fibers {
            return None;
        }
        let slot = reuse.unwrap_or(t.slots.len());
        // A reused slot keeps its finished occupant's generation (the ABA guard); a fresh slot is 0.
        let generation = if reuse.is_some() {
            t.slots[slot].own.generation()
        } else {
            0
        };
        let new_slot = Arc::new(FiberSlot {
            own: Ownership::new_owned_at(generation),
            running_on: AtomicU64::new(NOT_RUNNING),
            fiber: Mutex::new(Some(fiber)),
            // Fresh/reused: the shadow region starts empty (SP at the region base).
            shadow_sp: AtomicU64::new(fiber_region_base(slot)),
            func,
            sp,
        });
        if reuse.is_some() {
            t.free.pop();
            t.slots[slot] = new_slot;
        } else {
            t.slots.push(new_slot);
        }
        Some(fiber_handle(slot, generation))
    }

    /// Return a finished slot to the free list (recycling step 3); its generation was bumped by
    /// [`Ownership::finish`], so a later `cont.new` may reuse it ABA-safely.
    fn free_slot(&self, slot: usize) {
        self.lock().free.push(Reverse(slot));
    }

    /// Durable **thaw** re-seeding (DURABILITY.md §12.8 slice 3.3.3): re-create a frozen fiber at
    /// the next slot (dense, matching the freeze) as a fresh `OWNED` fiber — so a thaw `cont.resume`
    /// claims it (`Start`) and re-enters its entry under `REWINDING`, rebuilding then re-parking it —
    /// with its flattened shadow-SP restored so the swap re-points the active word to its region.
    fn seed_frozen(
        &self,
        fiber: Box<Fiber>,
        func: i32,
        sp: i64,
        shadow_sp: u64,
        generation: u64,
    ) -> usize {
        let mut t = self.lock();
        let slot = t.slots.len();
        t.slots.push(Arc::new(FiberSlot {
            // Re-seed at the freeze-time generation (recycling step 2), so a guest handle to a
            // recycled fiber still resolves; 0 for a non-recycled fiber (handle == slot).
            own: Ownership::new_owned_at(generation),
            running_on: AtomicU64::new(NOT_RUNNING),
            fiber: Mutex::new(Some(fiber)),
            shadow_sp: AtomicU64::new(shadow_sp),
            func,
            sp,
        }));
        slot
    }

    /// Resolve a (forgeable) handle: **masked** into the power-of-two-padded table (Spectre-safe,
    /// like `call_indirect` — and the same shape as the interp registry, so a forged handle now
    /// resolves over the same domain-wide namespace on both backends). Returns the **slot index** (for
    /// the per-context region + recycling) and the slot. Out of range ⇒ `None`.
    fn resolve(&self, handle: i64) -> Option<(usize, Arc<FiberSlot>)> {
        let t = self.lock();
        let mask = t.slots.len().next_power_of_two() - 1; // len 0 ⇒ mask 0 ⇒ slot 0, caught below
        let slot = (handle as u64 as usize) & mask; // the generation bits are above the slot mask
        if slot >= t.slots.len() {
            return None;
        }
        Some((slot, Arc::clone(&t.slots[slot])))
    }

    /// Run `f` over a **parked** fiber's live control-stack bytes `[ctx, top)` (W5 JIT/DWARF Stage 4c:
    /// the fiber-rooted backtrace). `None` if `handle` resolves to no live fiber, or to one currently
    /// *running* on a vCPU (whose saved `ctx` is stale and does not bound its frames). The slot's
    /// fiber lock is held across `f`, so the fiber cannot be resumed underneath it. Host-side tooling,
    /// off the runtime path (§2a) — only reads the parked stack.
    pub(crate) fn with_parked_stack<R>(
        &self,
        handle: i64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        let (_, slot) = self.resolve(handle)?;
        // Honor the generation ABA guard (recycling step 1/3): a stale handle whose slot has since
        // been recycled names no live fiber, so it gets no backtrace (mirrors `cont.resume`'s check).
        if slot.own.generation() != fiber_handle_generation(handle) {
            return None;
        }
        if slot.running_on.load(Ordering::Acquire) != NOT_RUNNING {
            return None; // a running fiber's `ctx` is the stale pre-resume context (see parked_extent)
        }
        let guard = slot.fiber.lock().unwrap_or_else(|e| e.into_inner());
        let fib = guard.as_ref()?;
        if fib.is_done() {
            return None;
        }
        let (lo, hi) = fib.parked_extent();
        let len = (hi as usize).checked_sub(lo as usize)?;
        // SAFETY: the fiber is parked and its lock is held, so `[lo, hi)` is a stable, mapped region
        // of its control stack (the live frames) for the duration of `f`; we only read it.
        let bytes = unsafe { std::slice::from_raw_parts(lo, len) };
        Some(f(bytes))
    }

    /// The handles of every **voluntarily-suspended** (`RUNNABLE`) fiber — the parked fibers the
    /// durable freeze driver flattens (DURABILITY.md §12.8). Collected once: flattening a fiber
    /// makes zero forward progress (its post-suspend poll unwinds immediately), so it spawns no new
    /// parked fibers, and it ends `FREE` — the set never grows.
    fn runnable_handles(&self) -> Vec<i64> {
        self.lock()
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.own.is_runnable())
            .map(|(i, s)| fiber_handle(i, s.own.generation())) // gen-carrying (recycling step 1/3)
            .collect()
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
    /// The OS-thread stack pointer captured at this vCPU's guest entry — the **high** bound for a
    /// `gc.roots` scan of the root computation's frames (its live region is `[root_low, root_entry_sp)`
    /// on the OS thread stack; the low bound is the current SP, or the outermost fiber's resumer SP
    /// when the collector runs inside a fiber). `0` until the entry path records it (and for vCPUs
    /// that never run guest code), which simply skips the root-frame scan.
    root_entry_sp: usize,
    /// **Durable** (DURABILITY.md §12.8): this run freezes/thaws fibers, so the resume swap keeps
    /// the active shadow-SP word pointing at the running context's region. `false` ⇒ the fiber
    /// runtime never touches the durable reserve (an ordinary run). Set per-run at entry.
    durable: bool,
    /// The guest window's host base (the durable swap reads/writes `window + SHADOW_SP_OFF`). Set
    /// per-run at entry (the window doesn't exist at construction). Unused unless `durable`.
    mem_base: u64,
    /// The root computation's saved shadow-SP (context 0) while it is parked resuming a fiber — the
    /// off-table root's slot in the per-context saved-SP table (a fiber's lives in its `FiberSlot`).
    root_shadow_sp: u64,
    /// The context currently running on this vCPU (`None` = the root, context 0; `Some` = a fiber).
    /// The resume swap reads it to know whose SP to save before switching in, and restores it after.
    cur_shadow: Option<Arc<FiberSlot>>,
    /// **Durable freeze residue** (slice 3.2/3.3.3): fibers this vCPU flattened during a freeze
    /// (`fiber_resume`'s `Complete` arm records each that unwound). Read back by `run_code_raw` into
    /// the durable entry's returned residue. Empty on a non-freeze run.
    frozen: Vec<crate::FrozenFiber>,
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
            root_entry_sp: 0,
            durable: false,
            mem_base: 0,
            root_shadow_sp: SHADOW_BASE,
            cur_shadow: None,
            frozen: Vec::new(),
        }
    }

    /// Record the finalized call-trampoline address (must be set before any fiber runs).
    pub(crate) fn set_call_tramp(&mut self, t: FiberCallTramp) {
        self.call_tramp = Some(t);
    }

    /// Arm the **durable** fiber-switch swap for this run (DURABILITY.md §12.8): record the window
    /// base and whether this is a durable run, so the resume swap can re-point the active shadow-SP
    /// word per context. Called by the entry path once the window is allocated. A non-durable run
    /// leaves `durable = false` and the fiber runtime never touches the reserve.
    pub(crate) fn set_durable_env(&mut self, mem_base: u64, durable: bool) {
        self.mem_base = mem_base;
        self.durable = durable;
    }

    /// Record the OS-thread stack pointer at this vCPU's guest entry — the high bound for the
    /// `gc.roots` root-frame scan (see [`FiberRuntime::root_entry_sp`]). Called by the entry path
    /// right before the guarded guest call.
    pub(crate) fn set_root_entry_sp(&mut self, sp: usize) {
        self.root_entry_sp = sp;
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

/// Build a suspended native fiber that, on first resume, runs guest `funcref(sp, arg)` through the
/// call-trampoline — resolving + type-checking the funcref against the function table on that first
/// resume, like the interpreter. Shared by `cont.new` ([`fiber_new`]) and durable thaw re-seeding
/// ([`SharedFiberTable::seed_frozen`]), so a thawed fiber re-enters its entry identically.
///
/// Returns `None` if the OS refuses the control-stack reservation — the caller turns it into a
/// `FiberFault`, never an abort, so a guest spawning many fibers can't crash the host (ISSUES.md I1).
///
/// # Safety
/// `fn_table_base`/`mem_base`/`trap_out`/`call_tramp` are this run's threaded context (live for the
/// fiber's lifetime); the body reads [`CURRENT_RT`] dynamically at each use (3c migration).
#[allow(clippy::too_many_arguments)]
unsafe fn make_fiber(
    funcref: i32,
    sp: u64,
    mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
    mask: u64,
    type_id: u32,
    call_tramp: FiberCallTramp,
) -> Option<Fiber> {
    Fiber::new(FIBER_STACK, move |y: &Yielder, arg: u64| -> u64 {
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
    })
}

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
) -> i64 {
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

    let fiber = match make_fiber(
        funcref,
        sp,
        mem_base,
        fn_table_base,
        trap_out,
        mask,
        type_id,
        call_tramp,
    ) {
        Some(f) => f,
        None => {
            // The OS refused the control-stack reservation — recoverable, not an abort (I1).
            fault(trap_out);
            return -1;
        }
    };

    let rt = &*rt;
    match rt.table.create(Box::new(fiber), funcref, sp as i64) {
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
    handle: i64,
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
    // Mid-run freeze trigger: count this `cont.resume` safepoint before the switch (mirroring the
    // interpreter's per-op tick) — on an armed durable run it may promote the window to UNWINDING, so
    // the resume's trailing poll begins the freeze.
    if (*rt).durable {
        window_tick_arm((*rt).mem_base);
    }
    // Phase 1: resolve + **claim** (D57): the slot must be this vCPU's (affinity, 3b-ii) and
    // `OWNED` — the `begin_owned` claim takes it to `RUNNING`, so a re-entrant resume (the fiber is
    // somewhere in a resume chain), a racing resume, or a finished fiber all *lose the claim* and
    // fault. No lock or `&mut` is held past this block; the `Arc` keeps the slot (and the boxed
    // fiber it owns) stable across the switch.
    let (slot_idx, slot, fib): (usize, Arc<FiberSlot>, *mut Fiber) = {
        let rt = &*rt;
        let Some((slot_idx, slot)) = rt.table.resolve(handle) else {
            fault(trap_out);
            *status_out = 1;
            return 0;
        };
        // **The 3c claim:** any vCPU may resume a fresh (`OWNED`) or suspended (`RUNNABLE`) fiber;
        // the acquire CAS arbitrates — exactly one racing claimant wins, and the winner
        // synchronizes-with the suspending thread's release, so the saved stack context is fully
        // visible even when *another* OS thread suspended it (the migration edge). The claim is
        // **generation-checked** (recycling step 1): the generation carried in the guest handle must
        // match the slot's, so a stale handle to a recycled slot's former occupant faults. All
        // generations are 0 until recycling is wired, so this equals the old `claim()` (handle == slot).
        if !slot.own.claim_gen(fiber_handle_generation(handle)) {
            fault(trap_out);
            *status_out = 1;
            return 0;
        }
        // Runtime single-owner assert (empirical net #3): a won claim must find the seam clear.
        // `RUNNING` slots are unclaimable, so a non-sentinel here means the protocol wiring is
        // broken — abort loudly rather than run one native stack on two threads. The swap is
        // **AcqRel** (and the clears below are **Release**) so this seam carries its *own*
        // happens-before — it observes the prior owner's clear independently of the `own` word, the
        // very thing it is meant to cross-check; a mis-wiring of `own` can no longer silence it in
        // lockstep.
        let prev = slot.running_on.swap(rt.me, Ordering::AcqRel);
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
        (slot_idx, slot, fib)
    };
    // Durable shadow-SP swap (DURABILITY.md §12.8, D-fiber-cont option A): `fiber_resume` brackets a
    // fiber's residency, so the whole swap lives here (no change to `fiber_suspend`). On entry save
    // the resumer's live shadow-SP and load this fiber's; on exit (the resume returns — the fiber
    // suspended *or* finished) save the fiber's and restore the resumer's. So a freeze that lands
    // while this fiber runs spills into *its* region, and the resumer resumes against its own.
    let (durable, mem_base) = {
        let rt = &*rt;
        (rt.durable, rt.mem_base)
    };
    let resumer = if durable {
        let rtm = &mut *rt;
        let resumer = rtm.cur_shadow.take(); // the context being suspended (None = root)
        let cur_sp = read_shadow_sp(mem_base);
        match &resumer {
            None => rtm.root_shadow_sp = cur_sp,
            Some(rs) => rs.shadow_sp.store(cur_sp, Ordering::Relaxed),
        }
        write_shadow_sp(mem_base, slot.shadow_sp.load(Ordering::Relaxed));
        rtm.cur_shadow = Some(Arc::clone(&slot));
        Some(resumer)
    } else {
        None
    };
    // Per-fiber trap attribution (DEBUGGING.md §5 W3 / §23-D57): publish this fiber as the one running
    // on this thread, so a trap inside the switch below is captured against *its* handle (the C
    // `trap_capture` reads the current-fiber TLS at the trap instant), and restore the resumer (root or
    // an outer fiber) when the resume returns. Stack-disciplined across nested resumes, mirroring the
    // durable shadow-SP bracket above — so a migrated fiber is named by identity, not by thread.
    let prev_fiber = svm_set_current_fiber(fiber_handle(slot_idx, slot.own.generation()));
    // Phase 2: the switch (may reenter the runtime) — no lock or `&mut` held; the claim makes
    // `*fib` exclusive to this vCPU. The same `svm-fiber` instruction sequence regardless of which
    // thread the fiber last ran on (see the module header's 3c soundness argument).
    let st = (*fib).resume(arg as u64);
    svm_set_current_fiber(prev_fiber);
    // Exit swap: back in the resumer (possibly on a different OS thread — re-read the runtime).
    // Save the fiber's now-current shadow-SP to its slot and restore the resumer's region.
    if let Some(resumer) = resumer {
        let rtm = &mut *current();
        slot.shadow_sp
            .store(read_shadow_sp(mem_base), Ordering::Relaxed);
        let restore = match &resumer {
            None => rtm.root_shadow_sp,
            Some(rs) => rs.shadow_sp.load(Ordering::Relaxed),
        };
        write_shadow_sp(mem_base, restore);
        rtm.cur_shadow = resumer;
    }
    // Phase 3: publish the fiber's new state (clearing the seam assert *before* republishing).
    match st {
        State::Yielded(v) => {
            // Voluntarily suspended: **publish to the pool** — claimable by any vCPU now, on any
            // thread (the migration point; release-pairs with the next claimant's acquire).
            slot.running_on.store(NOT_RUNNING, Ordering::Release);
            slot.own.suspend_to_pool();
            *status_out = 0;
            v as i64
        }
        State::Complete(v) => {
            // Durable freeze residue (DURABILITY.md §12.8 slice 3.2): this `Complete` is a freeze
            // **unwind** (not a genuine return) iff this is a durable UNWINDING run *and* the fiber
            // spilled into its shadow region (`shadow_sp` past its region base). That covers a fiber
            // unwound mid-resume-chain during the root run **and** a parked fiber driven by
            // `freeze_drive` — both Complete here. Record its residue so a thaw re-seeds it; a
            // genuine return (non-instrumented fiber, empty region) is left as an ordinary finish.
            let flat_sp = slot.shadow_sp.load(Ordering::Relaxed);
            if durable && flat_sp > fiber_region_base(slot_idx) && window_is_unwinding(mem_base) {
                (*current()).frozen.push(crate::FrozenFiber {
                    slot: slot_idx,
                    func: slot.func,
                    sp: slot.sp,
                    shadow_sp: flat_sp,
                    // Recorded before the `finish` below bumps it — the freeze-time generation, so a
                    // thaw re-seeds this (possibly recycled) fiber at the generation its handle carries.
                    generation: slot.own.generation(),
                });
            }
            // Drop the fiber (unmapping its stack) and free the slot — `finish` bumps the
            // generation, so any stale claim of this slot keeps failing; the slot returns to the free
            // list for recycling (step 3).
            slot.fiber.lock().unwrap_or_else(|e| e.into_inner()).take();
            slot.running_on.store(NOT_RUNNING, Ordering::Release);
            slot.own.finish();
            (*current()).table.free_slot(slot_idx);
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
    // Mid-run freeze trigger: count this `suspend` safepoint before the switch (mirroring the
    // interpreter's per-op tick). The promotion takes effect for the *resumer's* poll after this
    // fiber parks (suspend's own trailing poll is deferred to the fiber's next resume).
    if (*rt).durable {
        window_tick_arm((*rt).mem_base);
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

/// **Durable freeze driver** (DURABILITY.md §12.8 slice 3.3.2) — the JIT analogue of the
/// interpreter's `VCpu::freeze_drive`. Called once the root has unwound under `UNWINDING` (its
/// native stack drained into context 0's shadow region): flatten every still-**parked**
/// (`RUNNABLE`) fiber into *its own* shadow region so the window snapshot captures its continuation.
///
/// Each parked fiber is resumed (via the ordinary [`fiber_resume`] path, so the shadow-SP swap
/// points the active word at the fiber's region first): its `suspend` returns, and because the
/// transform places the poll **immediately** after, that poll fires before any guest code runs →
/// the fiber unwinds with **zero forward progress**, its guest function returns, and the `Fiber`
/// *completes* (the slot frees). The exit swap leaves the fiber's flattened shadow-SP in its
/// `FiberSlot` and restores the root's region, so the captured window is thaw-ready.
///
/// Runs **host-side** with `rt` (the root vCPU's runtime) published as [`CURRENT_RT`]. A flattening
/// fiber touches only the committed durable reserve (state word + its shadow region) — never guest
/// memory — so no guard page can fault; it is sound to run outside the §5 detect-and-kill guard.
/// Single-vCPU (slice 3.1/3.3); multi-vCPU stop-the-world quiesce is Phase 3.2.
///
/// Residue collection is **not** here: each flattened fiber's `fiber_resume` `Complete` arm records
/// it into `rt.frozen` (slice 3.2), the same path that captures a fiber unwound mid-resume-chain
/// during the root run — so [`take_frozen`] returns both. This driver only walks the parked set.
///
/// # Safety
/// `rt` is the live root runtime (its `mem_base`/`durable` armed for this run); `trap_out` is the
/// live trap cell. Called at a quiescent safepoint (the root returned), so the table is at rest.
pub(crate) unsafe fn freeze_drive(rt: *mut FiberRuntime, trap_out: u64) {
    // The entry path already published `rt` as CURRENT_RT and hasn't restored it yet, but set it
    // explicitly so the driver is correct independent of caller ordering.
    let prev = set_current(rt);
    let table = Arc::clone(&(*rt).table);
    let mut status: i64 = 0;
    for handle in table.runnable_handles() {
        // Resume under the in-progress UNWINDING state word: the fiber flattens itself, returns, and
        // its `Complete` arm records the residue into `rt.frozen`.
        fiber_resume(handle, 0, &mut status as *mut i64, trap_out);
    }
    set_current(prev);
}

/// Take the durable freeze residue this vCPU accumulated (slice 3.2/3.3.3) — the fibers flattened
/// during the root run (mid-resume-chain) and by [`freeze_drive`] (parked). Drains `rt.frozen`.
///
/// # Safety: `rt` is the live root runtime at a quiescent safepoint.
pub(crate) unsafe fn take_frozen(rt: *mut FiberRuntime) -> Vec<crate::FrozenFiber> {
    std::mem::take(&mut (*rt).frozen)
}

/// Durable **thaw** re-seeding (slice 3.3.3): re-create each frozen fiber in the run-shared table
/// before the root re-enters under `REWINDING`. Builds a fiber that re-enters its recorded entry
/// (`make_fiber`, the same path as `cont.new`) at its dense slot, with its flattened shadow-SP
/// restored. The seed is sorted/dense by slot, so handles match the freeze (`cont.resume k` resolves
/// the same fiber). `rt` supplies the per-run threaded context; `mem_base`/`fn_table_base`/`trap_out`
/// are this run's window/table/trap cell.
///
/// Returns `false` (after writing a `FiberFault` to `trap_out`) if the OS refuses a control-stack
/// reservation mid-seed — the caller skips the re-entry rather than aborting (ISSUES.md I1); `true`
/// when every frozen fiber was re-seeded.
///
/// # Safety
/// `rt` is the live runtime (its `call_tramp`/`mask`/`type_id` set); the addresses are this run's.
pub(crate) unsafe fn seed_frozen_fibers(
    rt: *mut FiberRuntime,
    seed: &[crate::FrozenFiber],
    mem_base: u64,
    fn_table_base: u64,
    trap_out: u64,
) -> bool {
    let r = &*rt;
    let (mask, type_id, call_tramp) = (
        r.fn_table_mask,
        r.fiber_type_id,
        r.call_tramp
            .expect("call-trampoline set before seeding thawed fibers"),
    );
    let mut seed = seed.to_vec();
    seed.sort_by_key(|f| f.slot);
    for (expected, f) in seed.iter().enumerate() {
        let Some(fiber) = make_fiber(
            f.func,
            f.sp as u64,
            mem_base,
            fn_table_base,
            trap_out,
            mask,
            type_id,
            call_tramp,
        ) else {
            // The OS refused a thaw control-stack reservation — recoverable, not an abort (I1).
            fault(trap_out);
            return false;
        };
        let got = r
            .table
            .seed_frozen(Box::new(fiber), f.func, f.sp, f.shadow_sp, f.generation);
        debug_assert_eq!(got, expected, "frozen fibers re-seed densely from slot 0");
        debug_assert_eq!(got, f.slot, "re-seeded slot matches the recorded handle");
    }
    true
}

/// This frame's stack pointer (approximately): the address of a local, slightly **below** the
/// caller's frame — a sound *low* bound for scanning the caller's live region upward. `inline(never)`
/// + `black_box` keep the probe from being optimized away or hoisted.
#[inline(never)]
fn current_sp() -> usize {
    let probe = 0usize;
    std::hint::black_box(&probe as *const usize as usize)
}

/// Scan raw native-stack words in `[low, high)` (host byte addresses), inserting every 8-byte word
/// that — after masking with `payload_mask` (`m = w & payload_mask`) — falls in the guest heap window
/// `[heap_lo, heap_hi)` into `out`. The **masked** value `m` is what's range-tested and inserted, so a
/// guest with tagged pointers (tag in the top byte) recovers the bare offset (`payload_mask = !0` is
/// the untagged case). Conservative: every aligned word is treated as a candidate root (spilled
/// pointers are 8-byte aligned on the control stack). An empty/inverted range scans nothing.
/// `payload_mask` is caller-validated to top-byte-strip only, so a host pointer stays large and is
/// excluded by the range test (no host-address leak — GC.md §3, §6).
///
/// # Safety
/// `[low, high)` must be a readable region of a *quiescent* native stack — a parked fiber's saved
/// extent, a paused resume-chain ancestor's stack, or the calling computation's own frames at a GC
/// safepoint. Reading a stack a *concurrent* thread is mutating (no stop-the-world) is a data race.
unsafe fn scan_words(
    low: usize,
    high: usize,
    heap_lo: u64,
    heap_hi: u64,
    payload_mask: u64,
    out: &mut BTreeSet<u64>,
) {
    let mut p = (low + 7) & !7usize; // first 8-aligned address at or above `low`
    while p.saturating_add(8) <= high {
        let w = (p as *const u64).read_unaligned();
        let m = w & payload_mask;
        if m >= heap_lo && m < heap_hi {
            out.insert(m);
        }
        p += 8;
    }
}

/// `gc.roots` thunk (§ GC.md §3/§6): a **conservative, ambient** root enumeration for the JIT. Walks
/// the live native control stacks of the running computation's fibers and reports every distinct
/// machine word that falls in the guest heap window `[heap_lo, heap_hi)` — the in-window words the
/// guest's own heap already encodes; out-of-window words (host return addresses, frame pointers, host
/// pointers) are filtered here and never cross the boundary. Writes the first `cap` candidates
/// (ascending, deduplicated) as little-endian `i64`s into guest memory at offset `buf`, and returns
/// the **total** found (the guest retries with a bigger buffer if it exceeds `cap`). This mirrors the
/// interpreter's `Inst::GcRoots` (which scans its reified `Value` frames) — soundness-equivalent
/// (a superset of the live roots), not word-for-word identical (GC.md §3.2: backends over-approximate
/// differently).
///
/// **Scan coverage (the "spilled-only" contract).** Every region scanned is one where roots are
/// already flushed to memory: (1) all **parked** fibers in the domain-shared table — `[ctx, top)`,
/// where the suspend spilled their callee-saved registers; (2) **running** resume-chain ancestors
/// (and the fiber calling this) — their whole usable stack `[usable_low, top)`, a sound superset;
/// (3) the **root computation's** frames on the OS thread stack — `[root_low, root_entry_sp)`. Live
/// roots a caller holds *only* in unspilled callee-saved registers of its own frame are out of scope
/// (documented; a register-flush shim is a future follow-up) — but the call boundary to this thunk
/// itself forces the `gc.roots` *caller* to spill, so its roots are covered.
///
/// **Concurrency (GC.md §3.3).** The scan is sound only at a stop-the-world safepoint — every other
/// vCPU parked, exactly as the interpreter scans the shared registry. We enforce the cheap sanity
/// check the contract permits: if any fiber is `RUNNING` on a vCPU *other than this one*, the caller
/// is not under STW, so the op **refuses** (`FiberFault`) rather than read a racing stack. The
/// current vCPU's own resume chain is quiescent while this synchronous thunk runs, so its `RUNNING`
/// fibers are scanned normally.
///
/// # Safety
/// `mem_base`/`mask`/`mapped`/`sub_base`/`trap_out` are the threaded guest-window context (as for
/// `cap.call`). `payload_mask` is the §GC tagged-pointer mask (distinct from the window-confinement
/// `mask`). The running vCPU's fiber runtime is read from [`CURRENT_RT`]; a null runtime, an
/// out-of-window `buf`, a non-STW call (a fiber running on another vCPU), or a `payload_mask` that
/// clears more than the top byte faults (`FiberFault`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn gc_roots(
    heap_lo: u64,
    heap_hi: u64,
    payload_mask: u64,
    buf: u64,
    cap: i64,
    mem_base: u64,
    mask: u64,
    mapped: u64,
    sub_base: u64,
    trap_out: u64,
) -> i64 {
    let rt = current();
    if rt.is_null() {
        fault(trap_out);
        return 0;
    }
    // Security: the payload mask may only clear the top byte (low 56 bits all-ones), else a host
    // pointer could be folded into the guest window and leak host-address bits past the range filter
    // (GC.md §3, §6). The verifier rejects a constant fold-down mask statically; this defends an
    // unverified module / non-constant mask, mirroring the interpreter's runtime check.
    if payload_mask | 0xFF00_0000_0000_0000 != u64::MAX {
        fault(trap_out);
        return 0;
    }
    let rt = &*rt;
    let mut roots: BTreeSet<u64> = BTreeSet::new();

    // (1)+(2) Every live fiber in the domain-shared table: a parked fiber's exact saved extent, or a
    // running fiber's whole usable stack (a sound superset — its exact SP is not known here). A
    // `FREE` (completed) slot holds no fiber (`take`n on finish) and is skipped.
    {
        let t = rt.table.lock();
        for slot in t.slots.iter() {
            let guard = slot.fiber.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(fib) = guard.as_ref() {
                let (lo, hi) = if slot.own.is_running() {
                    // §3.3 stop-the-world sanity check: a fiber running on *another* vCPU means the
                    // caller is **not** under STW — scanning its live, mutating native stack would be
                    // a data race. Refuse (fail closed with `FiberFault`) rather than read it. A
                    // fiber running on *this* vCPU is the caller's own resume chain (quiescent while
                    // this synchronous thunk runs) and is scanned as a superset.
                    if slot.running_on.load(Ordering::Relaxed) != rt.me {
                        fault(trap_out);
                        return 0;
                    }
                    fib.full_extent()
                } else {
                    fib.parked_extent()
                };
                scan_words(
                    lo as usize,
                    hi as usize,
                    heap_lo,
                    heap_hi,
                    payload_mask,
                    &mut roots,
                );
            }
        }
    }

    // (3) The root computation's frames on the OS thread stack, when recorded. Its low bound is the
    // current SP if `gc.roots` is called directly from the root (no fiber on this vCPU), else the
    // outermost fiber's resumer SP — the root's saved low-water mark at the point it resumed the
    // chain.
    if rt.root_entry_sp != 0 {
        let root_low = match rt.yielders.first() {
            None => current_sp(),
            Some(&y) => (*y).resumer_sp() as usize,
        };
        scan_words(
            root_low,
            rt.root_entry_sp,
            heap_lo,
            heap_hi,
            payload_mask,
            &mut roots,
        );
    }

    let total = roots.len();
    let nwrite = total.min(cap.max(0) as usize);
    if nwrite > 0 {
        // Confine the buffer offset exactly like the JIT's `mask_addr`: the writable region is the
        // *backed* window `[0, mapped)` (child-relative), physically at `mem_base + sub_base + off`.
        let masked = buf & mask;
        let nbytes = (nwrite as u64) * 8;
        if masked.checked_add(nbytes).is_none_or(|end| end > mapped) {
            fault(trap_out); // a forged / out-of-window buffer — like the interp's `MemoryFault`
            return 0;
        }
        let dst = (mem_base + sub_base + masked) as *mut u8;
        let mut off = 0usize;
        for w in roots.iter().take(nwrite) {
            std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), dst.add(off), 8);
            off += 8;
        }
    }
    total as i64
}

#[cfg(all(test, not(loom)))]
mod vcpu_ctx_tests {
    use super::{SharedFiberTable, MAX_FIBERS, MAX_SHADOW_CTX};

    // The durable vCPU-context allocator (slice 3.3): top-down reservation above the fiber pool, with
    // free-then-reuse (recycling) and a thaw-seed — the JIT mirror of the interp registry's `vcpu_mask`.
    #[test]
    fn reserve_is_top_down_and_recycles() {
        let t = SharedFiberTable::new(MAX_FIBERS);
        // Spawned vCPUs grow down from the top.
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX));
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX - 1));
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX - 2));
        // Freeing a context returns it to the pool; the next reserve reuses it (peak-concurrent bound).
        t.free_vcpu_context(MAX_SHADOW_CTX);
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX));

        // A thaw seed replaces the occupancy wholesale; reserve then avoids the seeded bit.
        t.seed_vcpu_mask(1 << (MAX_SHADOW_CTX - 1));
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX));
        assert_eq!(t.reserve_vcpu_context(), Some(MAX_SHADOW_CTX - 2)); // skips the seeded one
    }

    #[test]
    fn reserve_exhausts_cleanly() {
        let t = SharedFiberTable::new(MAX_FIBERS);
        // A fresh table has no fibers, so contexts 1..=MAX_SHADOW_CTX are all free.
        for _ in 0..MAX_SHADOW_CTX {
            assert!(t.reserve_vcpu_context().is_some());
        }
        assert_eq!(
            t.reserve_vcpu_context(),
            None,
            "the reserve is full once every context is live"
        );
    }
}
