//! Cranelift JIT backend (`DESIGN.md` §9, §18).
//!
//! Cranelift is the chosen codegen **by design** (§1): it is the security mechanism,
//! not a liability — we share Wasmtime's most security-critical component, so the
//! escape-TCB *delta* we own is just this CLIF generation plus the §4 masking
//! lowering. Correctness is established by **differential testing against the
//! reference interpreter** (§18, invariants I1/I4), the oracle in `svm-interp`.
//!
//! ## Status: the whole IR
//! Lowers every IR op: `i32`/`i64`/`f32`/`f64` consts, all integer and float
//! arithmetic/bitwise/shift/rotate/compare ops (incl. trapping `div`/`rem`),
//! `eqz`/`select`/`clz`/`ctz`/`popcnt`, every conversion (extend/wrap/demote/promote/
//! reinterpret, int↔float, saturating **and** trapping `trunc`), **loads/stores with
//! confinement masking** (I1), **indirect calls with function-table dispatch** (I2),
//! **`cap.call` through a host thunk** (§9, see [`CapThunk`]), and every terminator —
//! `br`/`br_if`/`br_table`/`return`/`unreachable` plus direct and indirect tail calls.
//!
//! ## Traps ([`JitOutcome`])
//! A trap is terminal (the guest domain is killed, §5 detect-and-kill), but the host
//! must *observe* it rather than crash. Arithmetic traps (div/rem-by-zero, trapping
//! `trunc`, `unreachable`, indirect-call type) are detected with **explicit checks that
//! store a [`TrapKind`] code and `return` early**; **memory faults** use a real
//! **`PROT_NONE` guard page + a SIGSEGV/SIGBUS handler** that unwinds the call as
//! [`TrapKind::MemoryFault`] (see `mem.rs` / `trap_shim.c`, unix). Either way the
//! *observable semantics* are identical, so the differential harness checks the JIT and
//! interpreter agree on traps too.
//!
//! ## The confinement lowering (§4, invariant I1, trap-confinement)
//! Every access **bounds-checks the final effective address** against the reserved
//! window (`addr + offset` vs the compile-time constant `reserved − offset − width`;
//! cold `MemoryFault` trap on failure — this is [`svm_mask::Window::checked`], the
//! isolated, separately-fuzzed spec), then **clamps** the address feeding the access
//! with `& (reserved − 1)` before adding the window base. The clamp is architecturally
//! a no-op past the check; on the *speculative* path it is a data dependency that
//! confines misspeculated OOB accesses to the window+guard — the Spectre-v1 hardening
//! (§4, D38). The window allocation carries a small guard margin so a base near the top
//! plus the access width never escapes the allocation (a real deployment uses guard
//! *pages* + a fault for the width overrun).
//!
//! **Check/clamp elision (§1a guard-when-bounded, D36–D38).** A conservative per-block
//! upper-bound analysis ([`ub_of`]) proves some effective addresses are *already* `< size`;
//! for those the check and clamp are both dropped ([`in_window`] / `mask_addr`'s `elide`) —
//! closing part of the gap to wasm32's free guard-page accesses (the proof is over data
//! dependencies, so it holds speculatively too). This is the subset of guard-when-bounded
//! that needs no guard region (it only elides *provably in-window* accesses, never relying
//! on a fault); the full wasm32-style large-guard version awaits real guard pages (§5). A
//! wrong bound would be a confinement escape, so the analysis is upper-bound-only
//! (unknown ⇒ check+clamp) and the elision is differentially guarded by the escape-oracle
//! (final-memory equality, §18).
//!
//! ## Indirect-call dispatch (§3c, invariant I2)
//! `call_indirect` masks the guest index into a host-owned, power-of-two-padded
//! function table, checks the slot's `type_id` against the call's signature (trap on
//! mismatch — a forged/wrong-type index is inert), and calls the slot's code pointer.
//!
//! ## Calling convention
//! All functions share a natural CLIF ABI `(mem_base, fn_table_base, params…) ->
//! (results…)` (the `tail` call conv, so `return_call` works); the two context
//! pointers are threaded through every call. The entry is wrapped in a fixed
//! buffer-ABI trampoline `fn(args: *const i64, results: *mut i64, mem_base: *mut u8,
//! fn_table_base: *const FnEntry)` so [`compile_and_run`] can call any arity from Rust.

use core::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{
    F32, F32X4, F64, F64X2, I16, I16X8, I32, I32X4, I64, I64X2, I8, I8X16,
};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp as ClifRmwOp, BlockArg, BlockCall, ConstantData, Endianness, Function,
    InstBuilder, JumpTableData, MemFlags, SigRef, SourceLoc, StackSlotData, StackSlotKind, Type,
    UserFuncName, Value, ValueLabel,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::LabelValueLoc;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use std::sync::Arc;
use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, ConvOp, Data, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncIdx,
    FuncType, Inst, IntTy, IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator, VBitBinOp,
    VCvtOp, VFCmpOp, VFloatBinOp, VFloatUnOp, VICmpOp, VIntBinOp, VIntUnOp, VNarrowOp, VPMinMaxOp,
    VSatBinOp, VShape, VShiftOp, ValType, DEFAULT_RESERVED_LOG2,
};

mod dwarf;
pub mod gdb; // W5 JIT/DWARF tier (Stage 2c): in-memory ELF wrapper + the GDB JIT registration interface
mod mem; // guest-window allocation + the §4/§5 guard-page / detect-and-kill handler // W5 JIT/DWARF tier: synthesize DWARF from the Stage 1 machine-address → source map

// JIT fiber runtime (§12): host-side fiber table + `extern "C"` thunks for `cont.new`/`resume`/
// `suspend`, on top of the `svm-fiber` stack-switch substrate. Available where `svm_fiber::supported()`.
#[cfg(fiber_rt)]
mod fiber_rt;

// JIT `setjmp`/`longjmp` runtime (LLVM.md §"JIT `longjmp`", Option B): a host-side `jmp_buf` table +
// `extern "C"` slot thunks, with libc `_setjmp`/`_longjmp` called inline from JITted code. Unix-only
// among the fiber_rt targets (`setjmp_rt` cfg); elsewhere the JIT keeps bailing `Unsupported` and the
// interpreters cover it.
#[cfg(setjmp_rt)]
mod setjmp_rt;

// 1:1 OS-thread executor for `thread.spawn`/`thread.join` + the `wait`/`notify` futex (§12): the VM
// exposes these as *primitives*, not a scheduler — a spawned vCPU is one real OS thread; any M:N model
// is built by the guest runtime over these + `cont.*` (D22: no built-in scheduler). The futex core is
// loom-verified. Available where `svm_fiber::supported()` (x86-64 unix).
#[cfg(fiber_rt)]
mod os_thread_rt;
// PROCESS.md S1b/S1c: the canonical-key futex region registry — `svm-run` records a §13 `map`'s pages
// (so the JIT futex thunks canonicalize `Backed` addresses) and purges them at teardown. Real-runtime
// only; the loom futex model has no regions (`futex_key_of` is `Anon`-only there).
#[cfg(not(loom))]
pub use os_thread_rt::{region_canon_forget_window, region_canon_record};

// §12 per-vCPU TLS register (`vcpu.tls.get`/`set`): one i64 per OS thread (a vCPU). Always compiled
// (substrate-independent), so a plain non-fiber root has a TLS word too.
mod vcpu_tls;

// §12.8 4A.5 durable-runtime-internal per-OS-thread shadow-region base (`durable.shadow_base`): the
// base of the region the running durable context spills into, so concurrent vCPUs each have their own
// per-context shadow-SP word (retiring the shared `SHADOW_SP_OFF`). Runtime-private (no guest setter).
mod durable_shadow;

// Migratable-fiber ownership protocol (D57 / DESIGN.md §23): the loom-verified single-owner atomic
// state machine that guarantees a stolen fiber is resumed by exactly one thread — the gating safety
// core of stackful work-stealing, proven in isolation before the runtime integration + cross-thread
// resume land. Pure atomics; not yet wired into the live runtime.
#[cfg(fiber_rt)]
mod fiber_registry;

// §14 nesting runtime: the host side of the `Instantiator` capability for the JIT — `instantiate`
// re-compiles a child confined to a sub-window (nesting cost paid at setup) and runs it over the
// parent's live window; `join` returns its result. Available where children can run (`fiber_rt`).
#[cfg(fiber_rt)]
mod instantiator_rt;

// The windows placeholder-window commit primitive, reused by `svm-run`'s Memory-cap backend (it
// commits/grows tail pages of this same window; a plain `VirtualAlloc(MEM_COMMIT)` cannot commit a
// placeholder reservation). See `mem::win_commit_rw`.
#[cfg(windows)]
pub use mem::win_commit_rw;

/// Whether this build's JIT lowers the §12 fiber/thread/futex ops (`cont.*`, `thread.*`,
/// `atomic.wait`/`notify`) instead of bailing [`JitError::Unsupported`]. True on the targets where
/// `svm-fiber` provides a real stack switch — the `fiber_rt` cfg derived in `build.rs`, kept in
/// lockstep with `svm_fiber::supported()`. Exposed so tests can assert the platform gating against the
/// single source of truth rather than re-deriving the target set.
pub const fn fiber_supported() -> bool {
    cfg!(fiber_rt)
}

/// Largest window the reference JIT will back with a host allocation. Real deployments
/// reserve a huge guard-paged virtual range (§4); for the differential harness we map
/// `1 << size_log2` bytes (+ a guard page on unix), so cap it.
const MAX_JIT_WINDOW_LOG2: u8 = 26; // 64 MiB (the backed `mapped` extent)

/// Largest **reserved** virtual range (the mask domain) the reference JIT will `mmap` per
/// window. The reservation is `PROT_NONE` + `MAP_NORESERVE`, so this is virtual address space,
/// not committed memory; `2^40` matches `DESIGN.md` §4's host-configurable example. A real host
/// chooses this per its VA budget; the reference just caps it so a fuzzed/oversized request
/// can't ask for an absurd reservation.
const MAX_JIT_RESERVED_LOG2: u8 = 40; // 1 TiB of reserved VA (lazy)

/// Escape-oracle snapshot span (the `_with_host` capture): byte-compare the low `SNAP_CAP` bytes of
/// the window across interp + JIT, *including* reserved-tail pages the guest grew via the Memory cap
/// (not just the backed prefix). Bounds the per-seed snapshot cost while covering a generous growth
/// region. **Must match `svm_interp`'s `SNAP_CAP`** so both backends snapshot the same span.
const SNAP_CAP: usize = 1 << 18; // 256 KiB

/// A function-table entry (§3c `FnEntry`): host-owned, guest-unwritable. `type_id`
/// identifies the signature (distinct-`FuncType` index); `code` is the finalized
/// function address. `call_indirect` masks the guest index into the table, checks
/// `type_id` (offset 0), then calls `code` (offset 8) — confinement at the use site (invariant I2).
///
/// The fields are **atomic** so a guest-driven `install`/`uninstall` (a host-side write) is sound
/// against a *concurrent* `call_indirect` from another thread (DESIGN.md §22 threaded install). The
/// `#[repr(C)]` layout (`type_id` @0, `code` @8) is exactly what [`indirect_dispatch`] bakes its
/// loads against, unchanged. Two distinct guarantees, both platform-uniform:
/// - **Visibility** (a synchronized reader sees a complete install) rides the **guest's own**
///   acquire/release — the worker only dispatches a slot it learned about through a guest atomic, so
///   the install's stores are ordered-before the worker's dispatch loads via that synchronization,
///   on weakly-ordered targets too. The dispatch's own plain loads need no acquire. `publish` still
///   stores `code` before `type_id` (release-ordered) as belt-and-suspenders for a reader that
///   happens to synchronize on `type_id` directly.
/// - **No torn pointer** (a *racy* reader never sees a half-written `code`): each field is a single
///   atomic word, so a concurrent dispatch reads either the old or the new value — never a wild
///   pointer. A racy reinstall is the guest's own bug and is contained (trap), never an escape.
#[repr(C)]
pub(crate) struct FnEntry {
    type_id: AtomicU32,
    _pad: u32,
    code: AtomicU64,
}

impl FnEntry {
    /// A real/installed entry.
    pub(crate) fn new(type_id: u32, code: u64) -> FnEntry {
        FnEntry {
            type_id: AtomicU32::new(type_id),
            _pad: 0,
            code: AtomicU64::new(code),
        }
    }
    /// A trapping padding entry (`call_indirect` here is inert).
    pub(crate) fn padding() -> FnEntry {
        FnEntry::new(PADDING_TYPE_ID, 0)
    }
    /// The slot's signature id (the host-side bookkeeping read; the hot dispatch read is in
    /// generated code). `Acquire` pairs with [`Self::publish`]'s release of `type_id`.
    pub(crate) fn type_id(&self) -> u32 {
        self.type_id.load(Ordering::Acquire)
    }
    /// The slot's code pointer (host-side bookkeeping read).
    pub(crate) fn code(&self) -> u64 {
        self.code.load(Ordering::Acquire)
    }
    /// **Publish** an installed function: `code` first, then `type_id` (release-ordered), so a
    /// concurrent reader that sees the new `type_id` also sees the new `code`.
    pub(crate) fn publish(&self, type_id: u32, code: u64) {
        self.code.store(code, Ordering::Release);
        self.type_id.store(type_id, Ordering::Release);
    }
    /// **Clear** to trapping padding: `type_id` (the ready field) first so a new dispatch traps
    /// promptly, then zero `code`.
    pub(crate) fn clear(&self) {
        self.type_id.store(PADDING_TYPE_ID, Ordering::Release);
        self.code.store(0, Ordering::Relaxed);
    }
}

/// `type_id` for a table slot that holds no function (the power-of-two padding) — it
/// matches no call site, so a forged index landing here always traps.
const PADDING_TYPE_ID: u32 = u32::MAX;
/// `type_id` a `call_indirect` uses when its signature matches no function in the
/// module: distinct from every real id and from the padding sentinel, so it always
/// traps (no function could satisfy it).
const NO_MATCH_TYPE_ID: u32 = u32::MAX - 1;

/// The distinct function signatures in a module; a function's (or call site's) type id
/// is an index into this list — structural equality, matching the interpreter's check.
fn distinct_types(funcs: &[Func]) -> Vec<FuncType> {
    let mut out: Vec<FuncType> = Vec::new();
    for f in funcs {
        let ft = FuncType {
            params: f.params.clone(),
            results: f.results.clone(),
        };
        if !out.contains(&ft) {
            out.push(ft);
        }
    }
    out
}

/// The type id of `ty` among `distinct`, or [`NO_MATCH_TYPE_ID`] if absent.
fn type_id_of(distinct: &[FuncType], ty: &FuncType) -> u32 {
    distinct
        .iter()
        .position(|t| t == ty)
        .map(|i| i as u32)
        .unwrap_or(NO_MATCH_TYPE_ID)
}

/// Intern `ty` into the **append-only** type-id registry (DESIGN.md §22: the per-domain id space
/// made incremental), returning its stable id. Soundness of the `call_indirect` dispatch check
/// reduces to this map being an *injection*, *stable over time* (an id never remaps — appends
/// only), and *total over participants* (every signature that can appear at a call site or in
/// a table slot is interned before any code referencing it is lowered) — then id-equality
/// coincides exactly with the interpreter's structural equality. The registry is consulted
/// only at compile/install time (inside a synchronous `cap.call`, guest suspended); compiled
/// code holds ids as immediates and never reads it at runtime.
fn intern_type(distinct: &mut Vec<FuncType>, ty: &FuncType) -> Result<u32, JitError> {
    if let Some(i) = distinct.iter().position(|t| t == ty) {
        return Ok(i as u32);
    }
    // Defensive: never collide with the `NO_MATCH_TYPE_ID` / `PADDING_TYPE_ID` sentinels
    // (unreachable in practice — it would take ~2^32 distinct signatures).
    if distinct.len() as u64 >= NO_MATCH_TYPE_ID as u64 {
        return Err(JitError::Unsupported("type-id registry full"));
    }
    distinct.push(ty.clone());
    Ok((distinct.len() - 1) as u32)
}

/// Intern every signature `funcs` can put into play for table dispatch: each function's own
/// signature (what a table slot holding it would carry) and every `call_indirect` /
/// `return_call_indirect` **site** signature (what the check compares against). Site
/// signatures matter: an id is baked into the call site as an immediate when the unit is
/// lowered, so a site whose signature is only defined by a *later* unit must already hold the
/// real id — interning up front keeps id-equality ≡ structural equality across units instead
/// of freezing a site to the always-trapping `NO_MATCH_TYPE_ID`.
fn intern_unit_sigs(distinct: &mut Vec<FuncType>, funcs: &[Func]) -> Result<(), JitError> {
    for f in funcs {
        intern_type(
            distinct,
            &FuncType {
                params: f.params.clone(),
                results: f.results.clone(),
            },
        )?;
        for b in &f.blocks {
            for i in &b.insts {
                if let Inst::CallIndirect { ty, .. } = i {
                    intern_type(distinct, ty)?;
                }
            }
            if let Terminator::ReturnCallIndirect { ty, .. } = &b.term {
                intern_type(distinct, ty)?;
            }
        }
    }
    Ok(())
}

/// Why the JIT could not compile (or run) a function. The integer slice rejects
/// anything it does not yet lower with [`JitError::Unsupported`]; the differential
/// harness treats that as "skip", not "fail".
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JitError {
    /// An instruction/terminator the current slice does not lower yet.
    Unsupported(&'static str),
    /// Structurally invalid (a verified module never hits this; defensive only).
    Malformed,
    /// Cranelift rejected the generated CLIF or failed to compile it.
    Backend(String),
}

/// What a JIT'd run produced: either the result slots, or a **trap** with a kind code.
///
/// A trap is terminal (the guest domain is killed, §5 "detect-and-kill") — this just
/// reports it to the host instead of aborting the process. The reference JIT detects
/// traps with **explicit checks that store a code and return early**; a production JIT
/// would instead take a hardware fault (guard page / `#DE`) caught by a signal handler
/// (§5). The *observable semantics* — which inputs trap, and the kind — are identical,
/// which is what the differential oracle checks.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JitOutcome {
    Returned(Vec<i64>),
    Trapped(TrapKind),
    /// The guest invoked the `Exit` capability with this code (§3e) — terminal, but not
    /// an error.
    Exited(i32),
}

/// Per-page protection to re-establish on a guest window before a run — the durable-restore
/// step (DURABILITY.md §12.3). One entry per [`DURABLE_SNAPSHOT_PAGE`]-byte page of the window's
/// backed prefix; pages beyond the prefix (or a `Rw` entry) are left at the default RW. Lets a
/// thawed guest fault on a restored `Ro`/`Unmapped` page exactly as the frozen one would,
/// matching the interpreter (`svm-interp`'s `apply_prots`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowProt {
    Rw,
    Ro,
    Unmapped,
}

/// The host-side residue of a fiber the durable freeze driver flattened (DURABILITY.md §12.8) — the
/// JIT mirror of `svm_interp::FrozenFiber`. Its continuation lives in its in-window shadow region;
/// this carries what a thaw must re-seed: the registry slot (= guest handle), entry funcref +
/// data-stack base (to re-enter it), and the flattened shadow-SP extent. A durable **freeze** run
/// returns one per flattened fiber; a **thaw** run is handed them back to re-create the fibers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrozenFiber {
    pub slot: usize,
    pub func: i32,
    pub sp: i64,
    pub shadow_sp: u64,
    /// The slot's generation at freeze (recycling step 2): re-seeded on thaw so a guest handle to a
    /// recycled fiber still resolves. 0 for a non-recycled fiber. Mirrors `svm_interp::FrozenFiber`
    /// (48-bit field — the `i64` handle's generation bits).
    pub generation: u64,
}

/// The host-side residue of a **spawned vCPU** (a `thread.spawn` child) flattened by a multi-vCPU
/// durable freeze (DURABILITY.md §12.8 slice 3.3) — the JIT mirror of `svm_interp::FrozenVCpu`. Its
/// continuation lives in its own in-window shadow region (`shadow_region_base(task)`); this carries
/// what a thaw must re-attach: its task id (= shadow context index + spawn order), entry function +
/// spawn args (to re-enter it), and the flattened shadow-SP extent. A multi-vCPU **freeze** returns
/// one per flattened child; a **thaw** is handed them back to re-create the children.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FrozenVCpu {
    pub task: usize,
    /// The task that **spawned** this child (slice 3.4: nested spawns) — `0` for the root's direct
    /// children, a child's task for a grandchild. Thaw rebuilds the per-parent join tables from this so
    /// a grandchild's reloaded handle resolves in its parent's table. Mirror of `svm_interp`'s field.
    pub parent_task: usize,
    pub func: i32,
    pub args: Vec<i64>,
    pub shadow_sp: u64,
    /// §12.8 4A.5 follow-up A: `Some(result)` for a **completed-but-unjoined** concurrent child — one
    /// that finished before the freeze point, so its `thread.join` result must survive in the snapshot
    /// (the host-side Done cell isn't captured otherwise). The thaw delivers this result into the
    /// spawner's join table **without re-running** the child (no double side effects). `None` for a
    /// normal frozen child (re-spawned + rewound on thaw); `shadow_sp`/`func`/`args` are then inert.
    pub completed_result: Option<i64>,
}

/// A §14 **nested-child** re-attach record captured by a durable freeze (DURABILITY.md §4, "JIT
/// parity") — the JIT mirror of `svm_interp::FrozenNested`. Unlike a `thread.spawn` child
/// ([`FrozenVCpu`], its own vCPU with an entry+args to re-spawn), a §14 child is a nested *domain*
/// whose whole continuation lives in its **carve** — a `2^size_log2` sub-window of the parent's window
/// at `carve_off` (already inside the frozen window image). This carries only what a thaw needs to
/// re-create the child domain around that carve and re-enter it under `REWINDING`. Same-module +
/// still-running only for now (the freeze-export slice); a separate-module child's module digest and a
/// completed-but-unjoined child's join result are follow-ups (see the interp's fuller record).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FrozenNested {
    /// The task that **instantiated** this child — `0` for the root's direct child; a grandchild
    /// carries its parent-child's task (depth-2, a follow-up). Mirror of [`FrozenVCpu::parent_task`];
    /// a thaw groups by it to re-attach parents before children.
    pub parent_task: usize,
    /// The parent's join-table slot for this child (the guest-held handle value).
    pub slot: usize,
    /// The carve's window-relative base (`sub_base`), inside the frozen window image.
    pub carve_off: u64,
    /// The carve's (= the child window's) size, `log2`.
    pub size_log2: u8,
    /// The child's entry function index into the parent's own table (same-module).
    pub entry: u32,
}

/// The durable snapshot's window-image page granularity (must match `svm-snapshot`'s `PAGE` /
/// `svm-interp`'s `DURABLE_SNAPSHOT_PAGE`): a restored protection map has one entry per this many
/// bytes. Host-page-independent for artifact portability; a 4 KiB codec page sits within one host
/// page, so protecting it protects (at most) its host page — exact on a 4 KiB-page host.
pub const DURABLE_SNAPSHOT_PAGE: usize = 4096;

/// Window offset of durable shadow **context 0** (the root vCPU's region base) — an *empty* shadow-SP
/// extent. Must match `svm-interp`'s / `fiber_rt`'s `SHADOW_BASE`; duplicated here (not under
/// `cfg(fiber_rt)`) for the durable run-state defaults. The cross-backend artifact-equality property
/// catches any drift.
const DURABLE_SHADOW_BASE: u64 = 64;

/// The trap kinds the JIT can raise (a subset of the interpreter's `Trap`), numbered to
/// match the codes the lowered checks / the host thunk store into the trap cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum TrapKind {
    DivByZero = 1,
    IntOverflow = 2,
    BadConversion = 3,
    Unreachable = 4,
    IndirectCallType = 5,
    /// Forged / closed / wrong-type capability handle (§3c).
    CapFault = 6,
    /// A guest memory access faulted into the window's guard region (§4/§5) — caught by
    /// the signal handler and turned into detect-and-kill. The masking lowering confines
    /// every access to `[0, size)`, so in practice this is a width-overrun at the very top
    /// of the window, or (defense-in-depth) a masking/elision bug that the guard caught.
    MemoryFault = 8,
    /// Forged / out-of-range / already-running / finished fiber handle, a bad fiber-entry funcref, a
    /// `suspend` with no running fiber, or a fiber-bomb (§12). Matches `Trap::FiberFault`.
    FiberFault = 9,
    /// Forged / out-of-range / already-joined thread handle, or a thread-bomb (§12). Matches
    /// `Trap::ThreadFault`.
    ThreadFault = 10,
    /// The host **interrupted** a runaway guest (§5 the fuel/epoch kill-path): a non-terminating
    /// guest is stopped because the host set the interrupt cell (e.g. a watchdog timer). The
    /// lowering polls that cell at loop back-edges and function entries and traps here. Matches the
    /// interpreter's `Trap::OutOfFuel` — both report "the host bounded this run".
    OutOfFuel = 11,
    /// A `longjmp` to a `jmp_buf` that was never `setjmp`'d (a stale/forged token) — caught by the
    /// host `setjmp` table's lookup before the (skipped) `_longjmp` (§3b totality). Matches the
    /// interpreters' `Trap::Malformed` for the same condition (LLVM.md §"JIT `longjmp`").
    SetjmpFault = 12,
    /// A guest **control-stack overflow** caught by the software stack-limit check the JIT emits in
    /// each prologue when `feature = "stack-check"` (the arena/software-guard fiber model, which drops
    /// the per-fiber hardware guard page). A function whose frame would grow the native stack past the
    /// running fiber's low bound traps here rather than corrupting an adjacent fiber's stack —
    /// detect-and-kill (§5). NOTE: on the default guard-page backend a *fiber* overflow is NOT a clean
    /// `MemoryFault` — the fault happens at stack exhaustion and the SIGSEGV handler (`SA_ONSTACK` set,
    /// but no `sigaltstack` installed) double-faults on the exhausted stack and kills the process. Only
    /// in-window memory faults (handler has ample stack) surface as `MemoryFault`. The software check
    /// here traps through `trap_out` with no signal, so it is the only path that catches fiber overflow
    /// survivably (see `svm-jit/STACK_GUARD_FLIP.md`, "sigaltstack finding").
    StackOverflow = 13,
}

/// Trap-cell code the host thunk stores for an `Exit` (the exit code rides in the high
/// 32 bits of the `i64` cell). Distinct from every [`TrapKind`].
pub const EXIT_CODE: u32 = 7;

impl TrapKind {
    fn from_code(c: u32) -> Option<TrapKind> {
        Some(match c {
            1 => TrapKind::DivByZero,
            2 => TrapKind::IntOverflow,
            3 => TrapKind::BadConversion,
            4 => TrapKind::Unreachable,
            5 => TrapKind::IndirectCallType,
            6 => TrapKind::CapFault,
            8 => TrapKind::MemoryFault,
            9 => TrapKind::FiberFault,
            10 => TrapKind::ThreadFault,
            11 => TrapKind::OutOfFuel,
            12 => TrapKind::SetjmpFault,
            13 => TrapKind::StackOverflow,
            _ => return None,
        })
    }
}

/// The host callback the JIT invokes for `cap.call` (§9's trampoline). The caller wires
/// it to its capability host; the JIT bakes the function + ctx addresses in as constants
/// and calls it. Scalars cross as `i64` slots (`i32` in the low bits), buffers as the
/// `(ptr, len)` window borrow. On return, `*trap_out` is `0` for success, a [`TrapKind`]
/// code for a trap, or `EXIT_CODE | (exit_code << 32)` for an `Exit`.
///
/// # Safety
/// `ctx` is the caller's host pointer; `args`/`results` point at `n_args`/`n_results`
/// `i64` slots; `[mem_base, mem_base+mem_size)` is the guest window's backed prefix
/// (`mem_base` null if none) and `mem_reserved` is the full reserved mask domain
/// (`>= mem_size`) the guest may `map`-grow into via the Memory cap (§3e/§4); `trap_out`
/// points at the live trap cell. All must outlive the call.
pub type CapThunk = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    mem_base: *mut u8,
    mem_size: u64,
    mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
);

/// The **devirtualized `cap.call` fast path** (§9 / D45). A `cap.call` to a statically-known
/// `(type_id, op)` normally goes through the generic [`CapThunk`]: it marshals args through a stack
/// buffer, passes a 12-wide ABI (incl. `n_args`/`n_res`/`type_id`/`op`), and the host dispatches on
/// `(type_id, op)` at runtime. A `FastCapResolver` lets an embedder hand the JIT a **specialized**
/// host function for a given `(type_id, op)`, which the JIT calls **register-to-register** — args and
/// the result in registers, no stack marshalling, no runtime dispatch. Resolution happens once at
/// **compile time** (the JIT calls the resolver during codegen and bakes the returned address); a
/// `null` return falls back to the generic thunk, so an embedder can fast-path only its hot ops.
///
/// The specialized function's ABI is, for a `cap.call` with `n_args` args and **one** result
/// (multi-result ops fall back to the generic thunk):
/// `unsafe extern "C" fn(ctx, mem_base, mem_size, handle: i32, trap_out: *mut i64, a0: i64, …, aN: i64) -> i64`
/// — `ctx`/`mem_base`/`mem_size`/`trap_out` are exactly as in [`CapThunk`]; the `handle` is the
/// resolved capability handle (the fn still does the authority check); each `ai` is the i'th argument
/// widened to its i64 slot (an i32/f32 in the low bits, an f64 bit-pattern); the i64 return is decoded
/// to the result type. A 0-result op returns an ignored 0. The fn signals a trap via `trap_out`
/// exactly like the generic thunk.
///
/// **The resolver MUST gate on `(n_args, n_res)`**: the JIT builds the call signature from the IR
/// `cap.call`'s arity, so a returned fn whose Rust signature has a *different* arity is a C-ABI
/// mismatch. A frontend may emit a `cap.call` to any `(type_id, op)` with any sig (the verifier checks
/// only `args.len() == sig.params.len()`, not that it matches the host op), so the resolver must return
/// a fn **only** when `(n_args, n_res)` equals that fn's own arity — otherwise `null` (the generic
/// slot-based path handles the odd arity safely). Types never mismatch (every arg is passed as an i64
/// register, the result decoded from i64), so only arity matters.
///
/// # Safety
/// The resolver and every function it returns must honour the ABI above (incl. the arity gate) and
/// stay valid for the run.
pub type FastCapResolver = unsafe extern "C" fn(
    type_id: u32,
    op: u32,
    n_args: u32,
    n_res: u32,
) -> *const core::ffi::c_void;

// §15 **spawn quota** — the single shared type lives in `svm-ir` (re-exported here and as
// `svm_interp::Quota`), so a powerbox embedder sets it once and it binds all three backends
// identically, with no facade conversion (Followup F6). NB the JIT's vCPU table is **cumulative** (a
// joined slot isn't freed), so here `max_vcpus` bounds *total* spawns over the run — stricter than the
// interpreter's concurrent-liveness cap, but containment holds either way.
pub use svm_ir::Quota;

/// A resolved §14 **`Module` grant** — raw views into host-owned storage (the powerbox's module
/// table), filled in by a [`ModuleResolver`]. The pointers must stay valid for the whole run (the
/// host's table is append-only and the host outlives the run — the same lifetime contract as the
/// `cap.call` ctx). `memory_log2 < 0` means the module declares no memory.
#[repr(C)]
pub struct ResolvedModule {
    pub funcs: *const Func,
    pub n_funcs: usize,
    pub memory_log2: i32,
    pub data: *const Data,
    pub n_data: usize,
}

/// The host callback the §14 nesting runtime uses to resolve a guest's **`Module` handle** to the
/// granted module's code/data (so a guest can only instantiate modules it was given). Returns
/// nonzero and fills `out` on success, `0` for a forged/closed/wrong handle. Deliberately a
/// *separate* callback from [`CapThunk`]: module resolution yields host pointers, which must never
/// be reachable from a guest-issued `cap.call` (the generic dispatch on a Module handle is an inert
/// `CapFault`) — only the host-side nesting runtime calls this.
///
/// # Safety
/// `ctx` is the same host pointer as the run's `cap_ctx`; `out` points at a writable
/// [`ResolvedModule`]. The filled views must outlive the run (see [`ResolvedModule`]).
pub type ModuleResolver =
    unsafe extern "C" fn(ctx: *mut core::ffi::c_void, handle: i32, out: *mut ResolvedModule) -> i32;

/// A §14 **granted child** powerbox built host-side (PROCESS.md S2 JIT parity): an opaque child
/// `Host` (`ctx`) holding an `Instantiator` + `AddressSpace` over the child's own window and the
/// parent's re-granted coordinate-free capability, plus the three entry-arg handles the child
/// receives (`Instantiator`, `AddressSpace`, grant). Filled by a [`GrantChildBuilder`]; `ctx` is
/// owned by the caller and must be freed with the paired [`GrantChildReleaser`] after the child runs.
#[repr(C)]
pub struct GrantChild {
    pub ctx: *mut core::ffi::c_void,
    pub inst_handle: i32,
    pub as_handle: i32,
    pub grant_handle: i32,
}

/// The host callback the §14 nesting runtime uses for **`instantiate_granted`** (Instantiator op 8):
/// re-grant one of the parent's own coordinate-free capabilities (`Stream`/`Exit`/`Clock`, named by
/// `grant_handle`) into a **fresh child powerbox** confined to `[0, child_size)`, so a JIT child can
/// do I/O instead of being born destitute. Returns nonzero and fills `out` on success, `0` for a
/// forged / non-copyable handle (an inert `CapFault`). Like [`ModuleResolver`], it is a *separate*
/// callback from [`CapThunk`]: it yields a host pointer (the child `Host`), which must never be
/// reachable from a guest-issued `cap.call`.
///
/// # Safety
/// `ctx` is the run's `cap_ctx` (the parent `Host`); `out` points at a writable [`GrantChild`]. The
/// filled `ctx` must stay valid until released with the paired [`GrantChildReleaser`].
pub type GrantChildBuilder = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    grant_handle: i32,
    child_size: u64,
    out: *mut GrantChild,
) -> i32;

/// The host callback for **`instantiate_named`** (Instantiator op 11): the multi-cap, by-name analog
/// of [`GrantChildBuilder`]. Reads `grants_n` 16-byte records `{name_off, name_len, handle, flags}` at
/// window-relative `grants_ptr` (bounded to `[0, mem_size)`), re-grants each record's copyable handle
/// into a fresh child powerbox **under its name** (the child finds them by `cap.self.resolve`), and
/// fills `out` (`grant_handle` unused). Returns nonzero on success; `0` with `*trap_out` set to a
/// `MemoryFault` (out-of-window record/name) or `CapFault` (non-UTF-8 name / forged / non-copyable
/// handle) — the whole spawn fails closed, matching the interpreter's op-11 path.
///
/// # Safety
/// `ctx` is the run's `cap_ctx` (the parent `Host`); `[mem_base, mem_base+mem_size)` is the parent's
/// mapped window; `out`/`trap_out` are writable. The filled `ctx` is released with the
/// [`GrantChildReleaser`] in the same [`GrantChildHooks`].
pub type GrantNamedChildBuilder = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    mem_base: *mut u8,
    mem_size: u64,
    grants_ptr: u64,
    grants_n: u64,
    child_size: u64,
    out: *mut GrantChild,
    trap_out: *mut i64,
) -> i32;

/// Free a child `Host` built by a [`GrantChildBuilder`] or [`GrantNamedChildBuilder`] — called once,
/// after the granted child has run and its outcome is stashed for `join`. Deliberately paired with the
/// builder (rather than, say, leaking the host for the run's lifetime like a `Module` grant) because a
/// shell respawning applets would otherwise accumulate one child `Host` per spawn.
///
/// # Safety
/// `ctx` is a [`GrantChild::ctx`] a builder returned and that has not yet been released.
pub type GrantChildReleaser = unsafe extern "C" fn(ctx: *mut core::ffi::c_void);

/// The host callbacks a granted §14 child needs — build its powerbox (positional [`GrantChildBuilder`]
/// for op 8, or by-name [`GrantNamedChildBuilder`] for op 11), then free it after the run. Threaded
/// through the compile pipeline next to [`ModuleResolver`]; `None` ⇒ `instantiate_granted` (op 8) and
/// `instantiate_named` (op 11) are inert `CapFault`s (a host that re-grants nothing), exactly as a
/// missing module resolver makes the module ops a `CapFault`.
#[derive(Clone, Copy)]
pub struct GrantChildHooks {
    pub build: GrantChildBuilder,
    pub build_named: GrantNamedChildBuilder,
    pub release: GrantChildReleaser,
    /// IMPORTS.md phase 3 / S2.1: bind a spawned child module's import manifest against its freshly
    /// built powerbox (`(parent_ctx, child_ctx, module_handle)`) — the JIT-side twin of the
    /// interpreter's inline `Host::bind_child_manifest` at spawn.
    pub bind_imports: ChildManifestBinder,
}

/// Bind a child module's import manifest against its built powerbox host — see
/// [`GrantChildHooks::bind_imports`].
pub type ChildManifestBinder = unsafe extern "C" fn(
    parent_ctx: *mut core::ffi::c_void,
    child_ctx: *mut core::ffi::c_void,
    module: i64,
);

/// §9/§12 async-ring host seam. The asynchronous `IoRing.submit_async` parks a vCPU on an in-window
/// futex completion **counter** and an offload-pool worker wakes it — but the pool lives in the
/// embedder's `Host` while the futex lives in the JIT's per-run `Domain`. This trait bridges them: the
/// run publishes its futex-`notify` into the `Host` (so a worker can wake the parked vCPU), and drains
/// the pool before the window/`Domain` are freed. `svm_run` supplies the `Host`-backed impl; a run
/// with no async ring passes `None` (then `submit_async` is an inert `-EINVAL` and the guest falls back
/// to the synchronous `submit`).
pub trait AsyncHostHooks {
    /// Install the futex wake hook — `notify(key, count)` wakes up to `count` vCPUs parked on the
    /// confined counter address `key`. Called once, after the thread `Domain` is up, before the guest
    /// runs.
    fn install_notify(&self, notify: std::sync::Arc<dyn Fn(u64, u32) + Send + Sync>);
    /// Drain the offload pool and drop the wake hook. Called after every vCPU is joined and before the
    /// window / `Domain` are freed, so no worker still holds those pointers.
    fn finish(&self);
}

/// The default thunk for [`compile_and_run`] (no host): an empty powerbox, so every
/// `cap.call` is inert — a `CapFault` — exactly like the interpreter's `run`.
unsafe extern "C" fn empty_cap_thunk(
    _ctx: *mut core::ffi::c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    _results: *mut i64,
    _n_results: u64,
    trap_out: *mut i64,
) {
    unsafe { *trap_out = TrapKind::CapFault as i64 };
}

/// The inert [`CapThunk`] (every `cap.call` is a `CapFault`) for callers with no host — the
/// long-lived [`CompiledModule::compile`] counterpart of [`compile_and_run`]'s empty powerbox.
/// Pass with a null `cap_ctx`.
pub const INERT_CAP_THUNK: CapThunk = empty_cap_thunk;

/// The CLIF type backing an IR value type.
fn clif_ty(t: ValType) -> Type {
    match t {
        ValType::I32 => I32,
        ValType::I64 => I64,
        ValType::F32 => F32,
        ValType::F64 => F64,
        // §17/D58: a `v128` SSA value is canonically held as `I8X16`; lane ops bitcast to the
        // shape-specific vector type and back.
        ValType::V128 => I8X16,
        // An opaque `ref` lowers exactly as `i64` (GC forward-compat reservation, GC.md §6).
        ValType::Ref => I64,
    }
}

/// The shape-specific CLIF vector type for a lane op (all 128-bit).
fn vec_ty(shape: VShape) -> Type {
    match shape {
        VShape::I8x16 => I8X16,
        VShape::I16x8 => I16X8,
        VShape::I32x4 => I32X4,
        VShape::I64x2 => I64X2,
        VShape::F32x4 => F32X4,
        VShape::F64x2 => F64X2,
    }
}

/// The CLIF **lane** type for a shape (the scalar a lane holds in CLIF).
fn lane_clif(shape: VShape) -> Type {
    match shape {
        VShape::I8x16 => I8,
        VShape::I16x8 => I16,
        VShape::I32x4 => I32,
        VShape::I64x2 => I64,
        VShape::F32x4 => F32,
        VShape::F64x2 => F64,
    }
}

/// Reinterpret a 128-bit vector value to another 128-bit vector type (a no-op bitcast,
/// little-endian lane order). Used to move between the canonical `I8X16` and a shape type.
fn vcast(b: &mut FunctionBuilder, v: Value, to: Type) -> Value {
    if b.func.dfg.value_type(v) == to {
        return v;
    }
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    b.ins().bitcast(to, mf, v)
}

/// The CLIF type for an integer-class IR type (operands to int↔float conversions).
fn int_clif_ty(t: IntTy) -> Type {
    match t {
        IntTy::I32 => I32,
        IntTy::I64 => I64,
    }
}

/// The CLIF type for a float-class IR type.
fn float_clif_ty(t: FloatTy) -> Type {
    match t {
        FloatTy::F32 => F32,
        FloatTy::F64 => F64,
    }
}

/// Compile the whole module and run `func` on slot-encoded `args` (each `i64` is one
/// parameter slot; `i32`/`f32` occupy the low 32 bits). Returns the result slots, or a
/// [`JitOutcome::Trapped`] if the run trapped. Intended for the differential harness.
///
/// All functions are compiled with a **natural CLIF ABI** — `(mem_base, fn_table_base,
/// trap_out, params…) -> (results…)` — so direct/indirect/tail calls are ordinary CLIF
/// calls; the entry is wrapped in a fixed buffer-ABI trampoline (any arity).
pub fn compile_and_run(m: &IrModule, func: FuncIdx, args: &[i64]) -> Result<JitOutcome, JitError> {
    // No host: an empty powerbox, so any `cap.call` is an inert CapFault (like `run`).
    compile_and_run_with_host(m, func, args, empty_cap_thunk, core::ptr::null_mut())
}

/// Compile `func` of `m` to a reusable [`CompiledModule`] with the default no-host policy (an empty
/// powerbox — any `cap.call` is an inert CapFault, exactly like [`compile_and_run`]), so the caller can
/// **compile once and [`CompiledModule::run`] many times** (DESIGN.md §22's long-lived split). The
/// one-shot [`compile_and_run`] recompiles the whole module on *every* call (~ms of Cranelift codegen);
/// a hot loop or a benchmark isolating per-iteration compute from compile jitter should compile here and
/// reuse the returned module. For `cap.call` dispatch to a real host, call [`CompiledModule::compile`]
/// directly with a thunk.
pub fn compile(m: &IrModule, func: FuncIdx) -> Result<CompiledModule, JitError> {
    CompiledModule::compile(
        m,
        func,
        empty_cap_thunk,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0, // natural (non-B2-reserved) function-table size
    )
}

/// Like [`compile_and_run`], but `cap.call`s dispatch through `cap_thunk` with the
/// caller's `cap_ctx` (the powerbox host). The thunk + ctx addresses are baked into the
/// compiled code as constants — valid because the module is compiled, run once, then
/// discarded here.
///
/// # Safety
/// `cap_thunk`/`cap_ctx` must stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_with_host(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<JitOutcome, JitError> {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed. Callers
    // wanting a specific reservation use the `_reserved` capture entry.
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        None,
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,             // no re-granted child powerbox (op 8 → CapFault)
        None,             // no kill-path armed (use `_interruptible` to arm one)
        None,             // no async ring
        None,             // no fast cap resolver (use `_fast` to supply one)
        Quota::default(), // no spawn quota (use a powerbox quota via svm-run)
    )?
    .0)
}

/// Like [`compile_and_run_with_host`], but also supply a [`FastCapResolver`] so hot `cap.call`s to
/// the resolver's known `(type_id, op)` pairs take the **devirtualized fast path** (register-to-
/// register, no stack marshalling, no runtime dispatch — §9 / D45). Calls the resolver doesn't claim
/// fall back to `cap_thunk`. This is the entry an embedder uses once it has specialized host functions
/// for its hot capabilities (the generic `cap_thunk` stays the correctness fallback).
///
/// # Safety
/// `cap_thunk`/`cap_ctx`/`fast_resolver` (and every function it returns) must stay valid for the call
/// and honour their respective ABIs.
pub fn compile_and_run_with_host_fast(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    fast_resolver: FastCapResolver,
    quota: Quota,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        None,
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        None, // no kill-path armed
        None, // no async ring
        Some(fast_resolver),
        quota,
    )?
    .0)
}

/// Like [`compile_and_run_with_host`], but **arm the §5 fuel/epoch kill-path**: the lowering polls
/// `*interrupt` at every loop back-edge and function entry, and traps [`TrapKind::OutOfFuel`] as
/// soon as the host stores a non-zero value there. This is how a host bounds a *runaway* JIT guest
/// (an infinite loop / unbounded recursion) — the interpreter has always had this via its fuel
/// counter; this gives the production backend the matching, **guest-undisableable** kill-path
/// (DESIGN §5 / preemption). The caller owns `interrupt` (typically an `Arc<AtomicU64>`) and sets it
/// from a watchdog timer, a cross-domain preemption, a signal handler, etc.
///
/// # Safety
/// `interrupt` must point at a live `AtomicU64` that outlives the call; `cap_thunk`/`cap_ctx` must
/// stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_with_host_interruptible(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    interrupt: *const AtomicU64,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        None,
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        Some(interrupt),
        None, // no async ring
        None, // no fast cap resolver
        Quota::default(),
    )?
    .0)
}

/// [`compile_and_run_with_host_interruptible`] + the §9/D45 [`FastCapResolver`]: the production run
/// path — a guest-undisableable kill-path **and** hot `cap.call`s devirtualized. The resolver's
/// unclaimed ops fall back to `cap_thunk` unchanged.
///
/// # Safety
/// As [`compile_and_run_with_host_interruptible`], plus `fast_resolver` (and every fn it returns) must
/// honour the [`FastCapResolver`] ABI and stay valid for the call.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_with_host_interruptible_fast(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    interrupt: *const AtomicU64,
    fast_resolver: FastCapResolver,
    quota: Quota,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        None,
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        Some(interrupt),
        None, // no async ring
        Some(fast_resolver),
        quota,
    )?
    .0)
}

/// Like [`compile_and_run`], but seed the guest window with `init_mem` (its low bytes) and
/// return the final window contents — the JIT side of the **escape-oracle** (§18). A
/// verified module that runs to completion must leave a window byte-identical to the
/// interpreter's [`svm_interp::run_capture`]; any divergence is a confinement/codegen
/// escape — a load/store whose effective address was not masked into `[0, size)`.
pub fn compile_and_run_capture(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed.
    compile_and_run_capture_reserved(m, func, args, init_mem, DEFAULT_RESERVED_LOG2)
}

/// Like [`compile_and_run_capture`], but with a host **reservation policy**: the window masks
/// into `[0, 2^reserved_log2)` (the mask domain) while only the declared `1 << size_log2` bytes
/// are backed — an access into the reserved-but-unmapped tail faults (§4 "guard-when-bounded";
/// detect-and-kill, §5). `reserved_log2` is raised to at least `size_log2` (so `0` ⇒ fully
/// mapped) and capped at the reference JIT's [`MAX_JIT_RESERVED_LOG2`]. This is the JIT side of
/// the escape-oracle under the decoupled `reserved`/`mapped` model; both backends must be driven
/// with the *same* `reserved_log2` to stay in differential lockstep.
pub fn compile_and_run_capture_reserved(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        empty_cap_thunk,
        core::ptr::null_mut(),
        Some(init_mem),
        reserved_log2,
        None,
        None,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        None, // no kill-path armed
        None, // no async ring
        None, // no fast cap resolver
        Quota::default(),
    )
}

/// Run the guest confined to a §14 **nested sub-window** `[base, base+size)` of a fully-backed
/// parent of `parent_bytes` (both seeded from / snapshotted into `init_mem`-sized buffers). The
/// module's declared memory is the *child* (`size = 1 << size_log2`); `base` must be size-aligned
/// and `base + size ≤ parent_bytes`. The masking lowering adds `base` to every confined address
/// (matching [`svm_mask::Window::sub`]), so this is the JIT side of the **sub-window escape-oracle**:
/// pair it with the interpreter's [`svm_interp::run_capture_sub`] and byte-compare the whole parent —
/// a verified guest must leave every byte *outside* `[base, base+size)` untouched (confinement) and
/// the slice itself byte-identical to the interpreter (codegen). `init_mem` seeds the parent's low
/// bytes; the returned `Vec` is the whole parent window.
pub fn compile_and_run_capture_sub(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    base: u64,
    parent_bytes: u64,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        empty_cap_thunk,
        core::ptr::null_mut(),
        Some(init_mem),
        0, // fully-mapped child (reserved == size); the parent is fully backed
        None,
        Some(SubWindow { base, parent_bytes }),
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        None, // no kill-path armed
        None, // no async ring
        None, // no fast cap resolver
        Quota::default(),
    )
}

/// [`compile_and_run_capture_reserved`] + a live powerbox: `cap.call`s dispatch through
/// `cap_thunk`/`cap_ctx` (so a granted handle takes its **success** path) *and* the final window
/// is captured for the escape-oracle. Pairs with the interpreter's
/// [`svm_interp::run_capture_reserved_with_host`] to byte-compare the effects of the §3e Memory
/// capability (`map`/`unmap`/`protect`) across both backends.
///
/// # Safety
/// `cap_thunk`/`cap_ctx` must stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_capture_reserved_with_host(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    compile_and_run_capture_reserved_with_host_ex(
        m,
        func,
        args,
        init_mem,
        reserved_log2,
        cap_thunk,
        cap_ctx,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
    )
}

/// [`compile_and_run_capture_reserved_with_host`] + a §14 **module resolver**: the host callback the
/// nesting runtime uses to resolve a guest's `Module` handle when it `instantiate`s a
/// **separate-module child** (the Instantiator's module ops). `None` ⇒ module ops are an inert
/// `CapFault` (same as a host that granted no modules).
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`]; `resolve_module` (with `cap_ctx`) must honour
/// the [`ModuleResolver`] contract — in particular the resolved views must outlive the run.
/// `grant_child` (PROCESS.md S2 JIT parity) supplies the `instantiate_granted` (op 8) child-powerbox
/// builder/releaser; `None` ⇒ op 8 is an inert `CapFault`. Both hooks share `cap_ctx` as their host.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_ex(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    resolve_module: Option<ModuleResolver>,
    grant_child: Option<GrantChildHooks>,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        Some(init_mem),
        reserved_log2,
        // Escape-oracle over the §1a growth path: snapshot the low `SNAP_CAP` bytes (not just the
        // backed prefix) so guest-grown / `unmap`-ed reserved-tail pages are byte-compared too.
        Some(SNAP_CAP),
        None,
        resolve_module,
        grant_child,
        None, // no kill-path armed (the differential oracle runs to completion)
        None, // no async ring
        None, // no fast cap resolver
        Quota::default(),
    )
}

/// [`compile_and_run_capture_reserved_with_host`] that first **re-establishes** a captured
/// per-page protection map on the window — the durable-restore step (DURABILITY.md §12.3): a
/// thawed guest faults on a restored `Ro`/`Unmapped` page exactly as the frozen one would,
/// matching `svm-interp`. `init_prots[i]` is the protection of the backed-prefix page at
/// `[i*DURABLE_SNAPSHOT_PAGE, …)`; pages beyond `init_prots` (or `Rw`) keep the default RW.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_prots(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None, // no sub-window
        None, // no module resolver
        None, // no interrupt
        None, // no fast cap resolver
        Quota::default(),
        0, // one-shot path: natural table size
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.run(args, Some(init_mem), Some(SNAP_CAP), None)
}

/// An async **freeze controller** (DURABILITY.md Phase-4 Slice A, 4A.3): a caller-owned handle a
/// *controller thread* uses to request a stop-the-world freeze of an in-flight durable JIT run. The
/// run publishes its live window base here once the window is mapped ([`CompiledModule::run`]);
/// [`Self::request_freeze`] then stores `UNWINDING` into the window's state word, which the guest's
/// loop-header back-edge poll (4A.1) observes at its next iteration and begins unwinding — closing
/// the R6 latency caveat for a poll-free compute loop with no deterministic countdown.
///
/// This is the real async trigger (vs. the deterministic `arm_freeze_after_backedges` test oracle,
/// which the interpreter keeps because its window is private/synchronous). The interpreter has no
/// equivalent — only the JIT's window is a shared mmap a second thread can reach.
///
/// # Lifetime contract
/// [`Self::request_freeze`] may be called **at most once**, concurrently with an in-flight run whose
/// loop does not terminate on its own before the request lands — so the window the published base
/// points at is provably still mapped when the store happens (the store is what ends the run). After
/// the run returns, the base is retired to a sentinel and `request_freeze` becomes a no-op. A request
/// that races a run which finishes first is therefore a safe no-op, not a use-after-free.
pub struct FreezeController {
    /// `0` = window not live yet (spin); `usize::MAX` = run ended (no-op); else the live window base.
    base: AtomicUsize,
}

impl FreezeController {
    /// A fresh controller, shareable with a run and a controller thread.
    pub fn new() -> Arc<Self> {
        Arc::new(FreezeController {
            base: AtomicUsize::new(0),
        })
    }

    /// Request a freeze: spin until the run publishes its window base, then store `UNWINDING` into the
    /// state word. A no-op if the run already ended (the base was retired). At most one call.
    pub fn request_freeze(&self) {
        loop {
            match self.base.load(Ordering::Acquire) {
                0 => std::hint::spin_loop(), // window not mapped yet
                usize::MAX => return,        // run ended before the request landed
                base => {
                    // STATE_OFF = 0, STATE_UNWINDING = 1 (must match `svm-interp`/`svm-durable`). An
                    // aligned atomic i32 store the guest's back-edge poll loads (defined under §12
                    // races); release-ordered after the run's acquire-published base.
                    // SAFETY: per the lifetime contract the window at `base` is mapped here (the run
                    // is blocked in its non-terminating loop until this store lands).
                    unsafe { (*(base as *const AtomicI32)).store(1, Ordering::Release) };
                    return;
                }
            }
        }
    }

    /// Run-side: publish the live window base (the run is now blocked in the guest).
    fn publish(&self, base: usize) {
        self.base.store(base, Ordering::Release);
    }

    /// Run-side: retire the base once the guest returns, so a late `request_freeze` is a no-op.
    fn retire(&self) {
        self.base.store(usize::MAX, Ordering::Release);
    }
}

/// [`compile_and_run_capture_reserved_with_host_prots`] for a **durable** run (DURABILITY.md §12.8):
/// arms the per-fiber shadow-SP swap so a freeze that lands while a fiber runs spills into that
/// fiber's own shadow region (D-fiber-cont option A), drives the freeze (flattening parked fibers),
/// and round-trips the fiber residue.
///
/// - **Freeze:** pass empty `init_prots` + empty `seed`; returns the [`FrozenFiber`] residue of every
///   fiber the driver flattened (for the snapshot's Section 2).
/// - **Thaw:** pass the captured page-protection map as `init_prots` and the frozen fibers as `seed`;
///   they are re-created in the fiber table before the `REWINDING` re-entry. Returns an empty residue.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_durable(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    seed: &[FrozenFiber],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<(JitOutcome, Vec<u8>, Vec<FrozenFiber>), JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.frozen_seed = seed.to_vec();
    cm.durable = true;
    let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP), None)?;
    Ok((outcome, win, std::mem::take(&mut cm.frozen_out)))
}

/// The durable-nested freeze/thaw entry's return: outcome, window, flattened fibers, and the §14
/// **nested-child** freeze residue (each a [`FrozenNested`]).
pub type DurableNestedOutcome = (JitOutcome, Vec<u8>, Vec<FrozenFiber>, Vec<FrozenNested>);

/// [`compile_and_run_capture_reserved_with_host_durable`] that ALSO returns the §14 **nested-child**
/// freeze residue (DURABILITY.md §4 "JIT parity") — one [`FrozenNested`] per child that unwound into
/// its carve under the freeze. Use for a durable domain that nests §14 children; the extra residue is
/// what a thaw re-attaches. A separate entry so the existing `_durable` callers are unaffected.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host_durable`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_durable_nested(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    seed: &[FrozenFiber],
    nested_seed: &[FrozenNested],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<DurableNestedOutcome, JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.frozen_seed = seed.to_vec();
    cm.frozen_nested_seed = nested_seed.to_vec();
    cm.durable = true;
    let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP), None)?;
    Ok((
        outcome,
        win,
        std::mem::take(&mut cm.frozen_out),
        std::mem::take(&mut cm.frozen_nested_out),
    ))
}

/// [`compile_and_run_capture_reserved_with_host_durable`] wired for an **async freeze** (Phase-4
/// Slice A, 4A.3): publishes the live window base into `freeze` so a controller thread can
/// [`FreezeController::request_freeze`] mid-run — the real bounded-latency stop-the-world trigger for
/// a poll-free compute loop (vs. the deterministic `arm_freeze_after_backedges` test oracle). The
/// guest observes the controller's `UNWINDING` write at its next loop-header back-edge poll and
/// unwinds, exactly as a freeze-from-start or armed freeze does — so the artifact round-trips
/// identically; only the *trigger* differs.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host_durable`]; additionally `freeze`'s lifetime
/// contract (call `request_freeze` at most once, concurrently with this run) must hold.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_durable_interruptible(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    seed: &[FrozenFiber],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    freeze: Arc<FreezeController>,
) -> Result<(JitOutcome, Vec<u8>, Vec<FrozenFiber>), JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.frozen_seed = seed.to_vec();
    cm.durable = true;
    cm.freeze_ctl = Some(freeze);
    let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP), None)?;
    Ok((outcome, win, std::mem::take(&mut cm.frozen_out)))
}

/// The result of a **multi-vCPU** durable freeze/thaw run (slice 3.3): `(outcome, window image,
/// flattened-fiber residue, spawned-vCPU residue, root vCPU's flattened shadow-SP extent)`. On a thaw
/// the two residue vectors are empty and the extent is inert.
pub type DurableMvOutcome = (JitOutcome, Vec<u8>, Vec<FrozenFiber>, Vec<FrozenVCpu>, u64);

/// [`compile_and_run_capture_reserved_with_host_durable`] for a **multi-vCPU** durable domain
/// (DURABILITY.md §12.8 slice 3.3) — the full freeze + thaw of a domain whose root has `thread.spawn`ed
/// children. A durable run is single-worker, so children run **inline** (deferred during a freeze until
/// the root unwinds; re-attached + run before the root re-enters on a thaw).
///
/// - **Freeze** (`vcpu_seed` empty): returns the flattened fibers, the spawned-vCPU residue (each
///   [`FrozenVCpu`]: entry func, `(sp, arg)` operands, flattened shadow-SP), **and the root vCPU's
///   flattened extent** (`root_sp`, reported separately because the shared active-SP word ends at the
///   last child's extent) — everything a snapshot needs to record the whole multi-vCPU domain.
/// - **Thaw** (`vcpu_seed` = the frozen children, `root_sp` = the root's restored extent): re-attaches
///   and runs the children, then re-enters the root under `REWINDING`. Returns empty residue.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host_durable`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_durable_mv(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    seed: &[FrozenFiber],
    vcpu_seed: &[FrozenVCpu],
    root_sp: u64,
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<DurableMvOutcome, JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.frozen_seed = seed.to_vec();
    cm.frozen_vcpu_seed = vcpu_seed.to_vec();
    cm.thaw_root_sp = root_sp;
    cm.durable = true;
    let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP), None)?;
    Ok((
        outcome,
        win,
        std::mem::take(&mut cm.frozen_out),
        std::mem::take(&mut cm.frozen_vcpus_out),
        cm.frozen_root_sp_out,
    ))
}

/// [`compile_and_run_capture_reserved_with_host_durable_mv`] for a **genuinely-concurrent** freeze
/// (DURABILITY.md §12.8 Phase 4 Slice A.5 stage ii): the root's `thread.spawn`ed children run as real
/// OS threads (not the single-worker deferred model), and a [`FreezeController::request_freeze`] makes
/// every context — root and children — self-unwind into its **own** per-context shadow-SP region
/// concurrently (lock-free, since the stage-i relocation gave each its own SP word). The coordinator
/// (root) joins the children via the existing `join_all` and then runs the unchanged freeze-drive +
/// snapshot. Residue is canonically sorted at serialize, so the (racy) quiesce order can't change the
/// artifact (§12.6). `request_freeze` may be called at most once, concurrently with this run.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host_durable_mv`]; additionally `freeze`'s lifetime
/// contract (call `request_freeze` at most once, concurrently with this run) must hold.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_durable_mv_interruptible(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    seed: &[FrozenFiber],
    vcpu_seed: &[FrozenVCpu],
    root_sp: u64,
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    freeze: Arc<FreezeController>,
) -> Result<DurableMvOutcome, JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )?;
    cm.restore_prots = init_prots.to_vec();
    cm.frozen_seed = seed.to_vec();
    cm.frozen_vcpu_seed = vcpu_seed.to_vec();
    cm.thaw_root_sp = root_sp;
    cm.durable = true;
    cm.concurrent_durable = true;
    cm.freeze_ctl = Some(freeze);
    let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP), None)?;
    Ok((
        outcome,
        win,
        std::mem::take(&mut cm.frozen_out),
        std::mem::take(&mut cm.frozen_vcpus_out),
        cm.frozen_root_sp_out,
    ))
}

/// [`compile_and_run_capture_reserved_with_host`] + the §9/§12 **async-ring host seam**
/// ([`AsyncHostHooks`]): wires this run's futex-`notify` into the embedder's `Host` so an offload-pool
/// worker can wake a vCPU parked in `IoRing.submit_async`, and drains the pool before teardown. Use it
/// (with `svm_run::HostAsyncHooks`) when the guest exercises the asynchronous ring; otherwise the plain
/// entry point leaves `submit_async` an inert `-EINVAL`.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`]; `hooks` must outlive the run and its `Host` must
/// be the same one `cap_ctx` points at.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_async(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    hooks: &dyn AsyncHostHooks,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        Some(init_mem),
        reserved_log2,
        Some(SNAP_CAP),
        None,
        None,
        None, // no re-granted child powerbox (op 8 → CapFault)
        None, // no kill-path armed
        Some(hooks),
        None, // no fast cap resolver
        Quota::default(),
    )
}

/// A §14 **nested sub-window**: run the guest confined to `[base, base+child_size)` of a
/// parent region of `parent_bytes` (both fully backed). The masking lowering adds `base` to
/// every confined address (matching [`svm_mask::Window::sub`]); `base == 0` is the ordinary
/// top-level window (the add is elided). The parent is seeded/snapshotted whole, so the
/// escape-oracle can assert the child only ever touched its own slice.
#[derive(Clone, Copy)]
pub struct SubWindow {
    pub base: u64,
    pub parent_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    init_mem: Option<&[u8]>,
    reserved_log2: u8,
    snapshot_cap: Option<usize>,
    sub: Option<SubWindow>,
    resolve_module: Option<ModuleResolver>,
    grant_child: Option<GrantChildHooks>,
    interrupt: Option<*const AtomicU64>,
    async_hooks: Option<&dyn AsyncHostHooks>,
    fast_resolver: Option<FastCapResolver>,
    quota: Quota,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    // The historical one-shot lifecycle, now compile → run over the long-lived split
    // (DESIGN.md §22): `CompiledModule` owns the `JITModule` for the whole run and the
    // executable memory is freed when it drops, after `run` returns — behavior-identical
    // to the old inline compile→run→drop.
    #[cfg_attr(not(fiber_rt), allow(unused_mut))]
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        sub,
        resolve_module,
        interrupt,
        fast_resolver,
        quota,
        0, // one-shot path: natural table size (no B2 reservation)
    )?;
    // PROCESS.md S2 (JIT parity): install the `instantiate_granted` (op 8) host callbacks into the
    // §14 nursery before the guest runs (the nursery only exists when the module holds an
    // `Instantiator`). `None` leaves op 8 an inert `CapFault`.
    #[cfg(fiber_rt)]
    if let Some(n) = &cm._nursery {
        n.set_grant_hooks(grant_child);
    }
    #[cfg(not(fiber_rt))]
    let _ = grant_child;
    cm.run(args, init_mem, snapshot_cap, async_hooks)
}

/// A [`JITModule`] whose memory is actually **released on drop**. cranelift-jit deliberately
/// leaks all code memory when a bare `JITModule` drops (its `Memory` `mem::forget`s every
/// allocation so stale `fn` pointers can never fault) — reclaiming it requires the explicit
/// `unsafe free_memory()`, which this crate never called. That leak is per *compile*, and the
/// [`ArenaMemoryProvider`] both compile paths install reserves **256 MiB per module** — which on
/// Windows is `MEM_RESERVE | MEM_COMMIT` (the region crate commits eagerly; see the note in
/// cranelift's `arena.rs`), i.e. every compile permanently charged 256 MiB against the system
/// commit limit. A compile-heavy process (the differential fuzz loops, `durable_jit`
/// freeze/thaw, a §14 nursery, a REPL) pinned the runner's commit ceiling within ~dozens of
/// compiles, after which *unrelated* heap allocations abort (`memory allocation of N bytes
/// failed` → `0xc0000409`) and window commits fail (`os error 1455`) — the ISSUES.md **I3**
/// Windows CI flake family. On unix, overcommit hid the same leak as unbounded VA growth.
///
/// Freeing on drop is sound because both owners ([`CompiledModule`], [`ChildCode`]) already pin
/// the lifetime contract "nothing that points into the code may outlive this struct": the field
/// is declared last, so runtimes/tables/trampolines drop first, and no fiber, thread, or
/// installed table entry survives the owner (documented on the structs).
struct OwnedJit(Option<JITModule>);

impl OwnedJit {
    fn new(m: JITModule) -> OwnedJit {
        OwnedJit(Some(m))
    }
}

impl core::ops::Deref for OwnedJit {
    type Target = JITModule;
    fn deref(&self) -> &JITModule {
        self.0.as_ref().expect("JITModule present until drop")
    }
}

impl core::ops::DerefMut for OwnedJit {
    fn deref_mut(&mut self) -> &mut JITModule {
        self.0.as_mut().expect("JITModule present until drop")
    }
}

impl Drop for OwnedJit {
    fn drop(&mut self) {
        if let Some(m) = self.0.take() {
            // SAFETY: per the owners' documented contract, by the time this field drops no guest
            // code from this module is executing and no `fn` pointer into it is ever called again.
            unsafe { m.free_memory() };
        }
    }
}

/// A compiled module whose executable code **outlives a single run** (DESIGN.md §22: the
/// long-lived `JITModule` split). Owns the [`JITModule`] (the executable memory lives until
/// drop), the power-of-two-padded function table, the entry's buffer-ABI trampoline, and the
/// §12/§14 runtimes whose addresses are baked into the code. Produced by
/// [`CompiledModule::compile`]; [`CompiledModule::run`] executes the entry over a fresh guest
/// window (allocated per run — the window base is threaded as a runtime argument, not baked).
///
/// [`CompiledModule::define_extra`] then declares + defines + finalizes **additional**
/// functions into the same live module — the enabling primitive for the guest-driven `Jit`
/// capability (DESIGN.md §22). Extra functions are *thunk-reachable only*: they are **never**
/// installed in the function table, so the table mask baked into every `call_indirect` site
/// (`fn_table_mask`) never changes — the escape-relevant dispatch is byte-identical to the
/// one-shot path.
/// One function produced by [`CompiledModule::define_extra`]: its buffer-ABI **trampoline**
/// (for `invoke` over a fresh/live window, any arity) and its natural-ABI **entry** + interned
/// **`type_id`** (for B2 [`CompiledModule::install`] into the `call_indirect` table). Pointers
/// are valid for the life of the `CompiledModule`.
#[derive(Clone, Copy)]
pub struct DefinedFn {
    pub tramp: *const u8,
    pub code: *const u8,
    pub type_id: u32,
}

/// `(func, block, inst) → index into the module's `debug_info.locs`` — the source-loc lookup the
/// JIT codegen consults to stamp each emitted op with a `cranelift SourceLoc` (W5 JIT/DWARF tier,
/// Stage 0). Built once per compile only when the module carries `-g` debug info.
type SrcLocMap = std::collections::HashMap<(u32, u32, u32), u32>;

/// Per-function captured `MachSrcLoc` ranges before address resolution: `(func index, [(start
/// offset, end offset, `debug_info.locs` index)])`, relative to the function's start until
/// `finalize` resolves the base address (W5 JIT/DWARF Stage 1).
type RawSrcLocs = Vec<(u32, Vec<(u32, u32, u32)>)>;

/// One source variable's machine-location ranges before address resolution (W5 JIT/DWARF Stage 3a):
/// `[(start offset, end offset, location)]`, relative to its function's start until `finalize`.
type VarLocRanges = Vec<(u32, u32, VarMachineLoc)>;

/// Per-function captured value-label ranges before address resolution (Stage 3a): `(func index,
/// [(value-label id, ranges)])`.
type RawVarLocs = Vec<(u32, Vec<(u32, VarLocRanges)>)>;

/// A machine-address → source range in finalized JIT code (W5 JIT/DWARF tier, Stage 1): the absolute
/// `[lo, hi)` code range a source position covers, resolved after `finalize` from Cranelift's
/// per-function `MachSrcLoc` ranges (relative offsets) + the function's finalized base address.
/// Strippable host-side tooling, untrusted-for-escape (§2a) — never read by running guest code.
#[derive(Clone, Copy, Debug)]
pub struct SrcRange {
    pub lo: u64,
    pub hi: u64,
    pub func: u32,
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

/// A symbolized JIT code address (W5 JIT/DWARF tier): the source position [`CompiledModule::
/// symbolize`] resolves a machine `pc` to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JitFrameLoc {
    pub func: u32,
    /// The source function name (`debug_info.func_names`), or `None` when the module carried no name
    /// for `func` — renderers fall back to `fn{func}`.
    pub func_name: Option<String>,
    pub file: String,
    pub line: u32,
    pub col: u32,
}

/// Symbolize a captured trap stack into source frames (§5 W3 Stage 1): the innermost frame from the
/// faulting `pc`, then one per **return address** in `rets` (the frame-pointer chain the guard
/// handler walked while the guest stack was intact), stopping at the first that isn't guest code per
/// `sym`. A return address points at the instruction *after* the call, so callers are symbolized at
/// `ret - 1` — landing inside the call instruction, the caller's real source position (the standard
/// backtrace adjustment); the innermost `pc` is the faulting instruction itself, symbolized
/// directly. Adjacent duplicate positions are collapsed, as in [`CompiledModule::fiber_backtrace`].
/// Pure (the handler already did the stack reads): the seed for [`CompiledModule::trap_backtrace`].
fn symbolize_capture(
    pc: usize,
    rets: &[usize],
    sym: impl Fn(usize) -> Option<JitFrameLoc>,
) -> Vec<JitFrameLoc> {
    let mut frames: Vec<JitFrameLoc> = Vec::new();
    if let Some(loc) = sym(pc) {
        frames.push(loc);
    }
    for &ret in rets {
        match sym(ret.wrapping_sub(1)) {
            Some(loc) => {
                if frames.last() != Some(&loc) {
                    frames.push(loc);
                }
            }
            None => break, // reached the host boundary (the outermost guest frame's caller) — stop.
        }
    }
    frames
}

#[cfg(test)]
mod trap_capture_tests {
    use super::{symbolize_capture, JitFrameLoc};

    fn loc(func: u32, line: u32) -> JitFrameLoc {
        JitFrameLoc {
            func,
            func_name: None,
            file: "f.c".into(),
            line,
            col: 0,
        }
    }

    #[test]
    fn symbolizes_pc_directly_callers_at_ret_minus_one_and_stops_at_host() {
        // pc → func2; ret0-1 → func1; ret1-1 → func0; ret2-1 → host (None, stop). A guest entry
        // is fed `pc` directly but callers via `ret - 1` (the byte inside the call instruction).
        let sym = |a: usize| match a {
            0x100 => Some(loc(2, 5)),  // pc (faulting instruction)
            0x199 => Some(loc(1, 10)), // ret0 (0x19a) - 1
            0x299 => Some(loc(0, 20)), // ret1 (0x29a) - 1
            _ => None,
        };
        let frames = symbolize_capture(0x100, &[0x19a, 0x29a, 0x999], sym);
        assert_eq!(
            frames,
            vec![loc(2, 5), loc(1, 10), loc(0, 20)],
            "three guest frames, host trimmed"
        );
    }

    #[test]
    fn collapses_adjacent_duplicate_positions() {
        // A recursive self-call: the same source position on consecutive frames collapses to one.
        let sym = |a: usize| match a {
            0x10 | 0x1f | 0x2f => Some(loc(0, 7)),
            _ => None,
        };
        let frames = symbolize_capture(0x10, &[0x20, 0x30, 0x40], sym);
        assert_eq!(
            frames,
            vec![loc(0, 7)],
            "duplicate adjacent positions collapse"
        );
    }
}

/// `(func, block) → [(block-local value index, value-label id)]` — the source variables to stamp
/// with a Cranelift `ValueLabel` during lowering (W5 JIT/DWARF Stage 3a). Built once per compile from
/// the §6 `debug_info.vars`' SSA-resident `VarLoc`s, only when the module carries `-g`. The label id
/// is the variable's index into [`CompiledModule::var_locs`]; the block-local value index maps onto
/// the JIT's per-block value map (see [`lower_block`]).
type VarLabelMap = std::collections::HashMap<(u32, u32), Vec<(u32, u32)>>;

/// Where a source variable's value physically lives over a machine-code range (W5 JIT/DWARF Stage
/// 3a) — Cranelift's `LabelValueLoc` translated to DWARF terms, the bridge to a `DW_AT_location`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarMachineLoc {
    /// In machine register `dwarf_reg` (a DWARF register number, via `map_regalloc_reg_to_dwarf`) —
    /// emits `DW_OP_regN` (Stage 3c).
    Reg(u16),
    /// At canonical-frame-address + `off` — emits `DW_OP_call_frame_cfa` + offset / `DW_OP_fbreg`.
    CfaOffset(i64),
}

/// One `[lo, hi)` absolute machine-code range over which a source variable lives in [`VarMachineLoc`]
/// (W5 JIT/DWARF Stage 3a) — one entry of a DWARF location list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarRange {
    pub lo: u64,
    pub hi: u64,
    pub loc: VarMachineLoc,
}

/// A source variable's machine-location list in finalized JIT code (W5 JIT/DWARF Stage 3a): the
/// `value_labels_ranges` Cranelift produced for the CLIF value that backs it, resolved to absolute
/// pcs. Empty `ranges` ⇒ the optimizer dropped the value everywhere (gdb will show `<optimized
/// out>`). Host-side tooling, off the runtime path (§2a).
#[derive(Clone, Debug)]
pub struct VarMachineInfo {
    pub func: u32,
    pub name: String,
    /// The variable's structured type as a `debug_info.types` index (Stage 3b), or `None` when the
    /// module carried only a render-name. The Stage 3c `DW_TAG_variable` points `DW_AT_type` at it.
    pub type_id: Option<u32>,
    pub ranges: Vec<VarRange>,
}

pub struct CompiledModule {
    /// The padded function table `call_indirect` dispatches through. Its address is threaded as
    /// a runtime argument (not baked), but running code reads it — boxed so it never moves, and
    /// declared before `module` so drop order matches the old `drop(fn_table); drop(module)`.
    fn_table: Box<[FnEntry]>,
    /// The entry's buffer-ABI trampoline (finalized code, owned by `module`).
    tramp_code: *const u8,
    /// The entry's parameter count — `run` rejects shorter `args` (the trampoline reads
    /// exactly this many slots).
    n_params: usize,
    /// The entry's result count (`run` sizes the result buffer).
    n_results: usize,
    /// The module's own function count — where the `fn_table`'s installable padding begins.
    /// `uninstall` refuses to clear a slot `< n_real_funcs` (a real module function), so a guest
    /// can only reclaim slots it `install`ed.
    n_real_funcs: usize,
    // --- the baked lowering environment, reused verbatim by `define_extra` so an extra
    // --- function compiles against the *same* constants as the parent (same confinement
    // --- mask, same cap thunk, same table mask — DESIGN.md §22 "vmctx sharing").
    distinct: Vec<FuncType>,
    cap: CapEnv,
    fiber: FiberEnv,
    thread: ThreadEnv,
    inst: InstEnv,
    setjmp: SetjmpEnv,
    mask: u64,
    cap_mapped: u64,
    sub_base: u64,
    epoch_addr: i64,
    /// The `call_indirect` index mask fixed at compile time (`next_pow2(n_funcs) - 1`) and baked
    /// into every call site. `define_extra` compiles new units against this same constant.
    fn_table_mask: u64,
    /// Monotonic counter for unique `declare_function` symbol names across `define_extra` calls.
    next_extra: usize,
    /// Cumulative machine-code bytes of every `define_extra`'d function + trampoline lowered over
    /// this module's life — a **byte-accurate** code-arena occupancy measure (the actual emitted
    /// code size, summed at finalize; the dominant term in arena consumption). Like `next_extra` it
    /// only grows, and restarts near zero in a freshly-compacted module. An embedder watermarks on
    /// [`Self::extra_byte_count`] to auto-compact (DESIGN.md §22).
    extra_bytes: usize,
    /// Finalized machine-code bytes of the **base module** (every function body lowered by
    /// [`CompiledModule::compile`], summed at define time from Cranelift's `code_buffer`). The
    /// analogue of [`Self::extra_bytes`] for the initial compile — a byte-accurate measure of the
    /// module's emitted code size, exposed via [`Self::code_byte_count`]. Excludes the small
    /// buffer-ABI trampoline and any later `define_extra` units (those are in `extra_bytes`).
    base_bytes: usize,
    /// The in-flight run's window fault range, published by `run_code_raw` for the duration of
    /// the guarded call so a mid-run [`Self::invoke_extra`] can arm its nested recovery.
    /// `None` ⇒ no run in flight (invoke is rejected).
    live_fault_range: Option<(usize, usize)>,
    /// The source backtrace of the most recent [`Self::run`] that **trapped** (§5 W3 Stage 1):
    /// innermost guest frame first, symbolized from the trap site the guard handler captured.
    /// Empty after a non-trapping run (or a trap with no usable frame). Read via
    /// [`Self::last_trap_backtrace`]; host-side, off the runtime path (§2a).
    last_trap_backtrace: Vec<JitFrameLoc>,
    /// The guest **fiber handle** running when the most recent [`Self::run`] trapped (§5 W3 / §23-D57
    /// per-fiber attribution): `Some(handle)` for a trap inside a fiber (named even if it had migrated
    /// across vCPU threads — captured at the trap instant, not inferred from the thread), `Some(-1)`
    /// for the root computation, `None` after a clean run. Read via [`Self::last_trap_fiber`].
    last_trap_fiber: Option<i64>,
    // --- the per-run window *plan* (the window itself is allocated in `run`; the mask and
    // --- extents are baked into the code, so they were fixed at compile time).
    win_mapped: usize,
    win_reserved: usize,
    win_size: usize,
    mem_size_log2: Option<u8>,
    /// Initialized data segments, owned so a run can seed a fresh window (the module may
    /// outlive the borrowed `IrModule`).
    data: Vec<Data>,
    /// Per-page protections to re-establish on the window before a run — the durable-restore
    /// step (DURABILITY.md §12.3). Empty ⇒ none (the common path); set per-run by `run_inner`.
    restore_prots: Vec<WindowProt>,
    /// This run is **durable** (DURABILITY.md §12.8): the fiber runtime keeps the active shadow-SP
    /// word pointing at the running context's region (swapped on every fiber switch). `false` (the
    /// default) ⇒ an ordinary run that never touches the durable reserve. Set per-run at entry.
    durable: bool,
    /// §12.8 4A.5 stage (ii): engage the **concurrent** durable path — children spawned during NORMAL
    /// run as real OS threads with their own reserved shadow contexts and self-unwind concurrently on a
    /// freeze (vs. the single-worker deferred model). Set by the concurrent multi-vCPU interruptible
    /// entry; `false` everywhere else, so those paths are byte-identical.
    concurrent_durable: bool,
    /// Durable **thaw** seed (slice 3.3.3): frozen fibers to re-create in the table before a
    /// `REWINDING` run, so a thaw `cont.resume` re-enters them. Empty for a freeze / ordinary run.
    frozen_seed: Vec<FrozenFiber>,
    /// Durable **freeze** residue (slice 3.3.3): the fibers the freeze driver flattened this run,
    /// read back by the durable entry point. Empty unless a freeze flattened fibers.
    frozen_out: Vec<FrozenFiber>,
    /// Durable **freeze** residue (slice 3.3): the spawned vCPUs that unwound under the freeze (each
    /// run inline single-worker by `thread.spawn`). Drained from the `Domain` after the root unwinds,
    /// read back by the durable entry point. Empty unless a freeze caught a live child.
    frozen_vcpus_out: Vec<FrozenVCpu>,
    /// Durable **freeze** residue (§4 "JIT parity"): the §14 **nested children** that unwound under the
    /// freeze, each a [`FrozenNested`] re-attach record. Drained from the run's `Nursery` after the root
    /// unwinds, read back by the durable-nested entry point. Empty unless a freeze caught a live §14
    /// child.
    frozen_nested_out: Vec<FrozenNested>,
    /// Durable **freeze** residue (slice 3.3): the **root** vCPU's flattened shadow-SP extent, captured
    /// after the freeze driver (before the children overwrite the shared active-SP word). The root's
    /// continuation rides in the window image like everyone's, but the active-SP word ends at the last
    /// child's extent, so the root's own extent is reported separately for a thaw to restore. `0` on a
    /// non-freeze run.
    frozen_root_sp_out: u64,
    /// Durable **thaw** seed (slice 3.3): the spawned vCPUs to re-attach + run before the root re-enters
    /// under `REWINDING`. Empty for a freeze / ordinary run.
    frozen_vcpu_seed: Vec<FrozenVCpu>,
    /// Durable **thaw** seed (§4 "JIT parity"): the §14 **nested children** to re-attach + rewind before
    /// the parent re-enters under `REWINDING`, so its re-executed `join` resolves. Empty otherwise.
    frozen_nested_seed: Vec<FrozenNested>,
    /// Durable **thaw** input (slice 3.3): the root vCPU's restored shadow-SP extent (from the
    /// artifact), set as the active word before the root rewinds. `SHADOW_BASE` (empty) otherwise.
    thaw_root_sp: u64,
    /// Async freeze controller (Phase-4 Slice A, 4A.3): if set, the run publishes its live window base
    /// here before the guarded call and retires it after, so a controller thread's `request_freeze`
    /// can write `UNWINDING` into the running window. `None` for every non-interruptible run.
    freeze_ctl: Option<Arc<FreezeController>>,
    // --- §12/§14 runtimes whose stable addresses are baked into the code; they must live
    // --- exactly as long as the code can run, i.e. as long as `module`.
    #[cfg(fiber_rt)]
    fiber_rt: Option<Box<fiber_rt::FiberRuntime>>,
    #[cfg(fiber_rt)]
    domain: Option<Box<os_thread_rt::Domain>>,
    /// Kept alive because its address is baked into the module's `Instantiator` cap.call sites.
    #[cfg(fiber_rt)]
    _nursery: Option<Box<instantiator_rt::Nursery>>,
    /// Kept alive because its address (`setjmp.rt_addr`) is baked into the module's `SetJmp`/`LongJmp`
    /// sites (LLVM.md §"JIT `longjmp`"). Holds the per-run host `jmp_buf` table.
    #[cfg(setjmp_rt)]
    _setjmp_rt: Option<Box<setjmp_rt::SetjmpRuntime>>,
    #[cfg(fiber_rt)]
    call_tramp: Option<fiber_rt::FiberCallTramp>,
    /// `(fiber_type_id, fiber_mask)` when the module uses `cont.*` — the per-vCPU fiber config
    /// spawned vCPUs build their runtimes from (`Domain::set_env`).
    #[cfg(fiber_rt)]
    fiber_cfg: Option<(u32, u64)>,
    /// The **domain-shared fiber table** (D57 3b-ii) the root's `fiber_rt` and every spawned
    /// vCPU's runtime are built over — one handle namespace + one §15 fiber quota per domain.
    #[cfg(fiber_rt)]
    fiber_table: Option<std::sync::Arc<fiber_rt::SharedFiberTable>>,
    /// Machine-address → source map for finalized code (W5 JIT/DWARF Stage 1), sorted by `lo`.
    /// Empty unless the module carried `-g` debug info. Host-side tooling, off the runtime path.
    src_ranges: Vec<SrcRange>,
    /// Source file paths (from `debug_info.files`), indexed by [`SrcRange::file`].
    src_files: Vec<String>,
    /// Source function names (`func → name`, from `debug_info.func_names`): for [`Self::symbolize`],
    /// the DWARF `DW_AT_name`, and kill messages. Empty unless the module carried `-g` names.
    func_names: std::collections::HashMap<u32, String>,
    /// Per-source-variable machine-location lists (W5 JIT/DWARF Stage 3a), resolved from Cranelift's
    /// `value_labels_ranges` after finalize. Empty unless the module carried `-g` vars. Host-side
    /// tooling, off the runtime path.
    var_locs: Vec<VarMachineInfo>,
    /// The §6 structured type graph (`debug_info.types`), emitted as `DW_TAG_*_type` DIEs (W5
    /// JIT/DWARF Stage 3b). Empty unless the module carried `-g` types. Host-side tooling.
    debug_types: Vec<svm_ir::TypeDef>,
    /// Owns the executable memory — the whole point of the long-lived split. Dropped last
    /// (declaration order), after everything that points into it — and the drop **releases** the
    /// code arena back to the OS (see [`OwnedJit`]; a bare `JITModule` would leak it).
    module: OwnedJit,
}

impl CompiledModule {
    /// Symbolize a finalized-code machine address to its source position (W5 JIT/DWARF Stage 1), or
    /// `None` if `pc` is outside any source-mapped op (a non-`-g` build, a trampoline, a prologue
    /// gap). The [`SrcRange`] map is sorted and disjoint, so this is a binary search. Host-side
    /// tooling, off the running guest's path (§2a) — the seed for trap symbolization + DWARF emit.
    pub fn symbolize(&self, pc: usize) -> Option<JitFrameLoc> {
        let pc = pc as u64;
        let i = self.src_ranges.partition_point(|r| r.lo <= pc);
        let r = self.src_ranges[..i].last().filter(|r| pc < r.hi)?;
        Some(JitFrameLoc {
            func: r.func,
            func_name: self.func_names.get(&r.func).cloned(),
            file: self
                .src_files
                .get(r.file as usize)
                .cloned()
                .unwrap_or_default(),
            line: r.line,
            col: r.col,
        })
    }

    /// The finalized machine-address → source map (W5 JIT/DWARF Stage 1), sorted by address. Empty
    /// unless the module carried `-g`. For tooling/tests and the forthcoming DWARF line-program emit.
    pub fn src_ranges(&self) -> &[SrcRange] {
        &self.src_ranges
    }

    /// A backtrace of a **suspended fiber** (W5 JIT/DWARF Stage 4c — the W3-JIT fiber-rooted stack
    /// walk). Rooted at a fiber *handle* (§23/D57 migratable fibers, not the OS thread), it scans the
    /// parked fiber's live control stack `[ctx, top)` low→high (innermost frame first) and symbolizes
    /// every word that lands in this module's JIT'd guest code — each is a return address, i.e. a
    /// guest call frame. A conservative scan (like the GC-root walk) rather than a frame-pointer
    /// chase, so it is robust to the host runtime glue sitting between the guest frames and the
    /// suspend switch. Adjacent duplicate positions are collapsed. Empty if `handle` names no parked
    /// fiber or the module carried no `-g`. Host-side tooling, off the running guest's path (§2a).
    #[cfg(fiber_rt)]
    pub fn fiber_backtrace(&self, handle: i64) -> Vec<JitFrameLoc> {
        let Some(table) = self.fiber_table.as_ref() else {
            return Vec::new();
        };
        table
            .with_parked_stack(handle, |stack| {
                let mut frames: Vec<JitFrameLoc> = Vec::new();
                for w in stack.chunks_exact(8) {
                    let pc = u64::from_le_bytes(w.try_into().unwrap()) as usize;
                    if let Some(loc) = self.symbolize(pc) {
                        if frames.last() != Some(&loc) {
                            frames.push(loc);
                        }
                    }
                }
                frames
            })
            .unwrap_or_default()
    }

    /// Symbolize a JIT **trap** into a source backtrace (§5 W3 Stage 1): the innermost frame from the
    /// faulting `pc`, then one frame per guest caller from `rets` — the frame-pointer chain's return
    /// addresses the SIGSEGV/SIGBUS handler walked *while the guest stack was intact* and stashed
    /// (`mem::take_trap_frame`); the walk can't run on the host side because the trap unwinds back
    /// onto the same stack the host then reuses. Callers are symbolized at `ret - 1` (inside the call
    /// instruction); adjacent duplicate positions are collapsed, as in [`Self::fiber_backtrace`].
    /// Empty when the module carried no `-g` (nothing symbolizes) or the capture is host-only. The
    /// engine stashes the last run's trap in [`Self::last_trap_backtrace`]. Pure host-side tooling,
    /// off the running guest's path (§2a).
    pub fn trap_backtrace(&self, pc: usize, rets: &[usize]) -> Vec<JitFrameLoc> {
        symbolize_capture(pc, rets, |a| self.symbolize(a))
    }

    /// The source backtrace of this module's most recent [`Self::run`] that **trapped** (§5 W3 Stage
    /// 1): innermost guest frame first, symbolized from the trap site captured during that run. Empty
    /// when the last run returned/exited, the trap carried no usable frame (an explicit-check trap —
    /// Stage 2 — or a platform whose handler doesn't decode the fault frame), or the module carried
    /// no `-g`. Host-side, off the runtime path (§2a) — fold it into a kill/trap report.
    pub fn last_trap_backtrace(&self) -> &[JitFrameLoc] {
        &self.last_trap_backtrace
    }

    /// The guest **fiber** that was running when this module's most recent [`Self::run`] trapped (§5 W3
    /// / §23-D57 per-fiber attribution): `Some(handle)` for a trap inside a fiber, `Some(-1)` for the
    /// root computation (no fiber), `None` after a clean run. The handle is captured at the trap
    /// instant, so it names the right fiber even under work-stealing migration (where the fiber may have
    /// resumed on a different vCPU thread than it last suspended on). Pairs with
    /// [`Self::last_trap_backtrace`] to render *which fiber* trapped *where*; host-side, off the runtime
    /// path (§2a).
    pub fn last_trap_fiber(&self) -> Option<i64> {
        self.last_trap_fiber
    }

    /// The per-source-variable machine-location lists (W5 JIT/DWARF Stage 3a): for each `-g` source
    /// variable whose value the JIT could track, the `[lo, hi)` machine ranges over which it lives in
    /// a register or CFA-relative slot. The seed for the Stage 3c `DW_AT_location` loclists. Empty
    /// without `-g`; a variable with empty `ranges` was optimized out everywhere.
    pub fn var_locations(&self) -> &[VarMachineInfo] {
        &self.var_locs
    }

    /// The synthesized DWARF `.debug_line` section for this module's finalized code (W5 JIT/DWARF
    /// Stage 2): a line-number program over the JIT'd machine addresses, for gdb/lldb (via the
    /// forthcoming GDB JIT registration) to resolve addresses to source lines. Empty without `-g`.
    pub fn debug_line_section(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        dwarf::debug_line(&self.src_ranges, &self.src_files)
    }

    /// The synthesized `(.debug_info, .debug_abbrev, .debug_loc)` for this module's finalized code (W5
    /// JIT/DWARF Stages 2b/3b/3c): a compile unit holding the §6 `TypeDef` graph as `DW_TAG_*_type`
    /// DIEs (3b) and one `DW_TAG_subprogram` per function (2b), each carrying its source variables as
    /// `DW_TAG_variable` children with `.debug_loc` location lists (3c). All empty without `-g`.
    fn synth_debug_info(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        if self.src_ranges.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        dwarf::debug_info(
            &self.func_extents(),
            &self.debug_types,
            &self.var_locs,
            &self.func_names,
        )
    }

    /// The synthesized DWARF `(.debug_info, .debug_abbrev)` (W5 JIT/DWARF Stages 2b/3b/3c) — a compile
    /// unit with the §6 type DIEs, one `DW_TAG_subprogram` per function, and a `DW_TAG_variable` per
    /// tracked source variable (its `DW_AT_location` referring into [`Self::debug_loc_section`]). Both
    /// empty without `-g`.
    pub fn debug_info_sections(&self) -> (Vec<u8>, Vec<u8>) {
        let (info, abbrev, _) = self.synth_debug_info();
        (info, abbrev)
    }

    /// The synthesized DWARF `.debug_loc` (W5 JIT/DWARF Stage 3c): the variable location lists the
    /// `DW_TAG_variable` DIEs' `DW_AT_location`s point into — one `DW_OP_reg{N}` entry per
    /// register-resident machine range. Empty without `-g` (or when no variable is register-resident).
    pub fn debug_loc_section(&self) -> Vec<u8> {
        self.synth_debug_info().2
    }

    /// The synthesized DWARF `.debug_frame` CFI (W5 JIT/DWARF Stage 4a): one CIE with the JIT's
    /// uniform frame-pointer unwind rules + one FDE per function, so gdb can unwind a stopped JIT
    /// frame (`bt`) and compute the CFA the subprograms' `DW_AT_frame_base` refers to. Empty without
    /// `-g`.
    pub fn debug_frame_section(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        dwarf::debug_frame(&self.func_extents())
    }

    /// Each function's machine `[low_pc, high_pc)` extent, derived as the span (min `lo`, max `hi`)
    /// of its source-mapped ranges, sorted by start address. The basis for both the `.debug_info`
    /// subprograms (Stage 2b) and the ELF `.symtab`/`.text` extent (Stage 2c). Empty without `-g`.
    fn func_extents(&self) -> Vec<(u32, u64, u64)> {
        let mut per_fn: std::collections::BTreeMap<u32, (u64, u64)> =
            std::collections::BTreeMap::new();
        for r in &self.src_ranges {
            let e = per_fn.entry(r.func).or_insert((r.lo, r.hi));
            e.0 = e.0.min(r.lo);
            e.1 = e.1.max(r.hi);
        }
        let mut funcs: Vec<(u32, u64, u64)> = per_fn
            .into_iter()
            .map(|(f, (lo, hi))| (f, lo, hi))
            .collect();
        funcs.sort_by_key(|&(_, lo, _)| lo);
        funcs
    }

    /// The in-memory ELF object that wraps this module's finalized code + synthesized DWARF for the
    /// GDB JIT interface (W5 JIT/DWARF Stage 2c): an `SHT_NOBITS` `.text` at the live code address,
    /// the `.debug_line`/`.debug_info`/`.debug_abbrev` sections, and a `.symtab` naming each
    /// function. This is the `symfile` [`Self::register_with_gdb`] hands to gdb. Empty without `-g`.
    pub fn elf_object(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        let funcs = self.func_extents();
        let code_base = funcs.iter().map(|f| f.1).min().unwrap_or(0);
        let code_end = funcs.iter().map(|f| f.2).max().unwrap_or(0);
        let (info, abbrev, loc) = self.synth_debug_info();
        let line = self.debug_line_section();
        let frame = self.debug_frame_section();
        gdb::build_elf(
            code_base,
            code_end.saturating_sub(code_base),
            &funcs,
            &self.func_names,
            &info,
            &abbrev,
            &line,
            &loc,
            &frame,
        )
    }

    /// Register this module's code with a live gdb/lldb via the GDB JIT interface (W5 JIT/DWARF
    /// Stage 2c) — the **headline milestone**: with the returned guard held, gdb can bind a
    /// source-line breakpoint inside the JIT'd code and show the guest source frame. The returned
    /// [`gdb::GdbRegistration`] **unregisters on drop**; hold it as long as the code is debuggable.
    /// `None` for a non-`-g` module (nothing to register). Host-side tooling, off the runtime path.
    pub fn register_with_gdb(&self) -> Option<gdb::GdbRegistration> {
        let elf = self.elf_object();
        if elf.is_empty() {
            return None;
        }
        Some(gdb::GdbRegistration::register(elf))
    }

    /// Compile the whole module (the compile half of the old one-shot `compile_and_run*`):
    /// declare + define every function, the entry's buffer-ABI trampoline, finalize once, and
    /// build the function table. Everything *baked into code* — the confinement mask, the
    /// `cap.call` thunk/ctx, the runtime addresses, the table mask, the §5 interrupt cell — is
    /// fixed here; per-run state (the window, the trap cell) is supplied by [`Self::run`].
    ///
    /// # Safety-relevant contract (not `unsafe`, but load-bearing)
    /// `cap_thunk`/`cap_ctx`/`fast_resolver`/`interrupt` addresses are baked into the compiled
    /// code: they must stay valid for the **lifetime of this `CompiledModule`** (not just one
    /// run) and honour their respective ABIs — the same contract the one-shot entry points
    /// documented per call, stretched over the module's life.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        m: &IrModule,
        func: FuncIdx,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        reserved_log2: u8,
        sub: Option<SubWindow>,
        resolve_module: Option<ModuleResolver>,
        interrupt: Option<*const AtomicU64>,
        fast_resolver: Option<FastCapResolver>,
        quota: Quota,
        table_reserve_log2: u8,
    ) -> Result<CompiledModule, JitError> {
        let entry = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
        // The `call_indirect` function table is power-of-two padded; `table_reserve_log2`
        // (DESIGN.md §22) reserves a *larger* table than the module needs so `install` can
        // fill the padding slots without moving the Spectre-safe mask constant (which is baked
        // from this length into every call site). `0` ⇒ the natural `next_pow2(funcs.len())`,
        // i.e. behavior-identical to before B2.
        let table_len = (1usize << table_reserve_log2)
            .max(m.funcs.len().next_power_of_two())
            .max(1);
        // §5 fuel/epoch kill-path: the address of the host-owned interrupt cell the lowering polls at
        // loop back-edges + function entries. `0` when the caller armed no kill-path (then no checks are
        // emitted — guest code is byte-identical to before). The cell must outlive the module; the caller
        // owns it (e.g. an `Arc<AtomicU64>` a watchdog thread sets), so the baked address stays valid.
        let epoch_addr = interrupt.map_or(0, |p| p as i64);
        // Calls can reach any function, so every function must be lowerable.
        for f in &m.funcs {
            ensure_supported(f)?;
        }

        // Plan the guest window if the module declares memory (allocation happens per `run`):
        // `mapped` backed RW bytes inside a host-configured `reserved` virtual range whose
        // unmapped tail + guard page fault (§4). `mask` is the §4 confinement mask (`reserved − 1`,
        // the mask domain); `win_size` is the seed/snapshot extent (the parent for a sub-window);
        // `cap_mapped` is the child's backed `mapped` that cap-call buffer borrows bound against.
        // `sub_base` is the §14 sub-window offset the masking lowering adds (0 for a top-level
        // window). All of these are baked into the code, so they are fixed at compile time.
        let (win_mapped, win_reserved, mask, win_size, cap_mapped, sub_base): (
            usize,
            usize,
            u64,
            usize,
            u64,
            u64,
        ) = match m.memory {
            Some(mc) => {
                if mc.size_log2 > MAX_JIT_WINDOW_LOG2 {
                    return Err(JitError::Unsupported(
                        "window too large for the reference JIT",
                    ));
                }
                let mapped = 1usize << mc.size_log2;
                // Host reservation policy: at least `mapped` (fully mapped if `reserved_log2` is
                // smaller, e.g. 0), capped so the reference JIT's reservation stays sane.
                let reserved_log2 = reserved_log2.max(mc.size_log2).min(MAX_JIT_RESERVED_LOG2);
                match sub {
                    // §14 sub-window: a fully-backed parent of `parent_bytes`, with the child
                    // confined (mask = child `reserved − 1`) into the slice at `base`. The child's
                    // mask domain `[0, reserved)` plus `base` must fit in the parent — the
                    // verifier-bounded child size + host-chosen `base` guarantee it (the
                    // Instantiator will enforce this).
                    Some(sw) => {
                        let parent = sw.parent_bytes as usize;
                        (
                            parent,
                            parent,
                            (1u64 << reserved_log2) - 1,
                            parent,
                            mapped as u64,
                            sw.base,
                        )
                    }
                    None => {
                        let reserved = 1usize << reserved_log2;
                        (
                            mapped,
                            reserved,
                            (1u64 << reserved_log2) - 1,
                            mapped,
                            mapped as u64,
                            0,
                        )
                    }
                }
            }
            None => (0, 0, 0, 0, 0, 0),
        };

        let mut flags = settings::builder();
        // A JIT'd function is called directly, not relocated into a shared object.
        let _ = flags.set("is_pic", "false");
        // Cranelift's x64 `return_call` (tail calls, §3b) lowering requires frame pointers.
        let _ = flags.set("preserve_frame_pointers", "true");
        // Run Cranelift's mid-end optimizer (GVN/CSE, constant materialization, store-to-load
        // forwarding). Wasmtime defaults to this; without it (the prior default `none`) redundant
        // address computations weren't CSE'd and constants were pool loads. "SSA on the wire" (no SSA
        // reconstruction) keeps cold start ahead even with the optimizer on.
        let _ = flags.set("opt_level", "speed");
        // Pin `enable_probestack` OFF (its 0.132 default). The software stack-overflow guard
        // (`stack-check`) is sound *because* a large frame's `sub rsp` is a pure pointer move that
        // touches no pages before the entry-block check runs — a probestack that page-walked the frame
        // downward would touch below the fiber's low bound *before* the check (silent neighbour-slot
        // corruption under the arena, which has no guard page). Pinning it locks that escape-TCB
        // assumption against a future Cranelift default flip. See `emit_stack_check` / STACK_GUARD.md.
        let _ = flags.set("enable_probestack", "false");
        let isa = cranelift_native::builder()
            .map_err(|e| JitError::Backend(e.to_string()))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| JitError::Backend(e.to_string()))?;
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // Allocate code + read-only data (float constants, jump tables) from a single contiguous
        // arena rather than the default separate `mmap`s. cranelift's x64 `call`/rip-relative
        // loads use 32-bit PC-relative relocations (`X86CallPCRel4`/`X86PCRel4`); with independent
        // mmaps, ASLR can place code and rodata > 2 GiB apart, overflowing the i32 offset (a
        // `compiled_blob.rs` panic) — intermittent, and only on large modules with rodata (e.g.
        // a whole UI library). A 256 MiB reserved arena (VA only, committed on demand) keeps every
        // segment in range. Reserve falls back to the default provider if it cannot map.
        if let Ok(arena) = cranelift_jit::ArenaMemoryProvider::new_with_size(256 << 20) {
            builder.memory_provider(Box::new(arena));
        }
        let mut module = JITModule::new(builder);

        // Declare every function (natural ABI) up front so calls can reference any of them.
        let ids: Vec<_> = m
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let sig = natural_sig(&mut module, f);
                module
                    .declare_function(&format!("f{i}"), Linkage::Local, &sig)
                    .map_err(|e| JitError::Backend(e.to_string()))
            })
            .collect::<Result<_, _>>()?;

        // Distinct signatures give each function (and call site) a structural type id, the
        // basis for the `call_indirect` type check (matching the interpreter). Function
        // signatures come first (`distinct_types`, ids identical to the historical one-shot
        // compile — the fn-table and `fiber_type_id` depend on those positions), then every
        // call-site signature is interned after them. Today a site whose signature matches no
        // function traps either way (a fresh id ≥ the function-sig count matches no table
        // entry, exactly like `NO_MATCH_TYPE_ID`) — but interning it now means a *later*
        // `define_extra`/install of a function with that signature can satisfy the site,
        // keeping id-equality ≡ structural equality across units (DESIGN.md §22).
        let mut distinct = distinct_types(&m.funcs);
        intern_unit_sigs(&mut distinct, &m.funcs)?;
        let distinct = distinct;

        // The host thunk + ctx addresses, baked into `cap.call` sites as constants.
        let cap = CapEnv {
            thunk_addr: cap_thunk as usize as i64,
            ctx_addr: cap_ctx as usize as i64,
            fast_resolver,
        };

        // §12 fibers: if the module uses `cont.*`, stand up a host fiber runtime (its address baked into
        // the `cont.*` sites). The `FiberRuntime` box is created now so its address is stable; the
        // call-trampoline address is filled in after finalize. On targets without stack-switch support,
        // `ensure_supported` has already rejected the fiber ops, so this stays `null`.
        #[cfg(fiber_rt)]
        let uses_fibers = module_uses_fibers(m);
        #[cfg(fiber_rt)]
        let uses_threads = module_uses_threads(m);
        // Per-run fiber constants (the §12 fiber entry type id + the function-table mask), used to build
        // fiber runtimes — one standalone, or one per vCPU when threads use `cont.*`.
        #[cfg(fiber_rt)]
        let fiber_type_id = type_id_of(&distinct, &fiber_func_type());
        #[cfg(fiber_rt)]
        let fiber_mask = (m.funcs.len().next_power_of_two() as u64) - 1;
        // Fibers + threads compose via a per-vCPU fiber runtime (execution context), published through a
        // thread-local, all over **one domain-shared fiber table** (D57 3b-ii: a unified handle
        // namespace + a per-domain §15 quota, matching the interpreter's run-shared registry). This is
        // the *root* vCPU's runtime (the one running `main` on the caller's thread); each spawned vCPU
        // builds its own over the same table from `fiber_cfg` (`os_thread_rt`). Created whenever the
        // module uses `cont.*` **or** `thread.spawn` — a threaded module needs the table for the
        // durable vCPU-context allocator (slice 3.3), even when it uses no fibers (the table is then
        // dormant: a fiber-free module never resumes a fiber, so the root's runtime just routes the
        // durable shadow-SP swap).
        #[cfg(fiber_rt)]
        let fiber_table: Option<std::sync::Arc<fiber_rt::SharedFiberTable>> =
            if uses_fibers || uses_threads {
                Some(std::sync::Arc::new(fiber_rt::SharedFiberTable::new(
                    quota.max_fibers,
                )))
            } else {
                None
            };
        // The *root* vCPU's fiber runtime is built only when the module actually uses `cont.*` — a
        // fiber-free module (even a threaded one) never resumes a fiber, so it needs no execution
        // context, and the durable shadow-SP word it does use is driven by the instrumented IR
        // directly (the multi-vCPU deferred/thaw spawn paths go through the `Domain`, not this
        // runtime). The *table* is still created for `uses_threads` above (the durable vCPU-context
        // allocator); only the per-vCPU runtime is fiber-gated, so a thread-only run publishes no
        // `CURRENT_RT` and allocates no idle runtime (matching the pre-slice-3.3 behavior).
        #[cfg(fiber_rt)]
        let mut fiber_rt: Option<Box<fiber_rt::FiberRuntime>> = if uses_fibers {
            fiber_table.as_ref().map(|t| {
                Box::new(fiber_rt::FiberRuntime::new(
                    std::sync::Arc::clone(t),
                    fiber_type_id,
                    fiber_mask,
                ))
            })
        } else {
            None
        };
        // The `cont.*` thunk addresses (the runtime itself is found via a thread-local at call time).
        #[cfg(fiber_rt)]
        let fiber = if uses_fibers {
            FiberEnv {
                new_thunk: fiber_rt::fiber_new as *const () as i64,
                resume_thunk: fiber_rt::fiber_resume as *const () as i64,
                suspend_thunk: fiber_rt::fiber_suspend as *const () as i64,
                // The register-flush trampoline (not `gc_roots` directly): it spills the callee-saved
                // registers before the scan so a guest root parked in one is captured (see its docs).
                roots_thunk: fiber_rt::svm_gc_roots_flush as *const () as i64,
            }
        } else {
            FiberEnv::null()
        };
        #[cfg(not(fiber_rt))]
        let fiber = FiberEnv::null();

        // §12 threads: stand up the 1:1 OS-thread executor `Domain` whose stable address is baked into the
        // `thread.*` sites. It owns no scheduling policy — `thread.spawn` launches a real OS thread (the
        // guest builds any M:N model itself, D22). The per-run `Env` (call-trampoline, window, trap cell)
        // is supplied after finalize via `set_env`; the address is stable now.
        #[cfg(fiber_rt)]
        let domain: Option<Box<os_thread_rt::Domain>> = if uses_threads {
            Some(Box::new(os_thread_rt::Domain::new(quota.max_vcpus)))
        } else {
            None
        };
        #[cfg(fiber_rt)]
        let thread = if let Some(d) = &domain {
            ThreadEnv {
                sched_addr: (&**d as *const os_thread_rt::Domain) as i64,
                spawn_thunk: os_thread_rt::thread_spawn as *const () as i64,
                join_thunk: os_thread_rt::thread_join as *const () as i64,
                wait_thunk: os_thread_rt::thread_wait as *const () as i64,
                notify_thunk: os_thread_rt::thread_notify as *const () as i64,
            }
        } else {
            ThreadEnv::null()
        };
        #[cfg(not(fiber_rt))]
        let thread = ThreadEnv::null();

        // §14 nesting: if the module holds an `Instantiator` (a `cap.call` to iface 6), stand up the
        // per-run `Nursery` whose stable address is baked into those sites. `instantiate` re-compiles a
        // child confined to a sub-window and runs it over this window (its detect-and-kill fault range is
        // supplied post-finalize via `set_env`, like the thread `Domain`). A child runs synchronously
        // today, so the nursery is touched only on the calling thread.
        #[cfg(fiber_rt)]
        let nursery: Option<Box<instantiator_rt::Nursery>> = if module_uses_instantiator(m) {
            Some(Box::new(instantiator_rt::Nursery::new(
                m.funcs.clone().into(),
                cap_thunk,
                cap_ctx,
                resolve_module,
                epoch_addr as usize, // §5: nested JIT children poll the parent's kill-path cell too
                0, // §4 depth-2: the **root** nursery's task id; its direct children get `parent_task = 0`
                std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1)), // next child task = 1
                std::sync::Arc::new(std::sync::Mutex::new(Vec::new())), // the subtree's shared residue sink
            )))
        } else {
            None
        };
        #[cfg(fiber_rt)]
        let inst = if let Some(n) = &nursery {
            InstEnv {
                nursery_addr: (&**n as *const instantiator_rt::Nursery) as i64,
                instantiate_thunk: instantiator_rt::instantiate as *const () as i64,
                join_thunk: instantiator_rt::join as *const () as i64,
                coro_spawn_thunk: instantiator_rt::coro_spawn as *const () as i64,
                coro_resume_thunk: instantiator_rt::coro_resume as *const () as i64,
                poll_thunk: instantiator_rt::poll as *const () as i64,
                detach_thunk: instantiator_rt::detach as *const () as i64,
                kill_thunk: instantiator_rt::kill as *const () as i64,
                instantiate_granted_thunk: instantiator_rt::instantiate_granted as *const () as i64,
                instantiate_named_thunk: instantiator_rt::instantiate_named as *const () as i64,
                instantiate_module_named_thunk: instantiator_rt::instantiate_module_named
                    as *const () as i64,
            }
        } else {
            InstEnv::null()
        };
        #[cfg(not(fiber_rt))]
        let inst = InstEnv::null();

        // `setjmp`/`longjmp` (LLVM.md §"JIT `longjmp`", Option B): when the module uses them, stand up
        // the per-run host `jmp_buf` table whose stable address is baked into the `SetJmp`/`LongJmp`
        // sites. Owned by the `CompiledModule` (`_setjmp_rt`), so it outlives every (re-)run. Unix-only
        // (the `setjmp_rt` cfg); elsewhere `ensure_supported` has already rejected the ops.
        #[cfg(setjmp_rt)]
        let uses_setjmp = module_uses_setjmp(m);
        // Fail-closed (guard against cross-stack corruption): the per-run `jmp_buf` table is keyed by
        // the guest buffer address and shared across the run's native stacks, so a module that mixes
        // `setjmp` with fibers/threads could let one stack's `setjmp` overwrite another's saved native
        // SP (a `longjmp` would then restore into the wrong stack). The on-ramp never emits that combo,
        // but a hand-crafted IR module could — so decline the JIT and let the interpreters (which key
        // `setjmp_points` per-vCPU) cover it. Per-fiber JIT keying is a documented follow-on.
        #[cfg(setjmp_rt)]
        if uses_setjmp && (uses_fibers || uses_threads) {
            return Err(JitError::Unsupported(
                "setjmp/longjmp combined with fibers/threads is not supported on the JIT yet",
            ));
        }
        #[cfg(setjmp_rt)]
        let setjmp_runtime: Option<Box<setjmp_rt::SetjmpRuntime>> = if uses_setjmp {
            Some(Box::new(setjmp_rt::SetjmpRuntime::new()))
        } else {
            None
        };
        #[cfg(setjmp_rt)]
        let setjmp = if let Some(r) = &setjmp_runtime {
            SetjmpEnv {
                rt_addr: (&**r as *const setjmp_rt::SetjmpRuntime) as i64,
                slot_thunk: setjmp_rt::rt_setjmp_slot as *const () as i64,
                lookup_thunk: setjmp_rt::rt_setjmp_lookup as *const () as i64,
                setjmp_addr: setjmp_rt::setjmp_addr(),
                longjmp_addr: setjmp_rt::longjmp_addr(),
            }
        } else {
            SetjmpEnv::null()
        };
        #[cfg(not(setjmp_rt))]
        let setjmp = SetjmpEnv::null();

        // W5 JIT/DWARF tier (Stages 0–1): when the module carries `-g` debug info, build the
        // `(func,block,inst) → debug_info.locs` index lookup that stamps each op with a `SourceLoc`,
        // and capture each function's `MachSrcLoc` ranges (relative offsets) to resolve to a
        // machine-address → source map after finalize. Off the runtime path; absent ⇒ no-op.
        let srcloc_map: Option<SrcLocMap> = m.debug_info.as_ref().map(|di| {
            di.locs
                .iter()
                .enumerate()
                .map(|(i, l)| ((l.func, l.block, l.inst), i as u32))
                .collect()
        });
        let mut raw_srclocs: RawSrcLocs = Vec::new();

        // W5 JIT/DWARF Stage 3a: assign each SSA-resident source variable a `ValueLabel` and record
        // the `(func, block) → [(block-local value, label)]` points `lower_block` stamps. Only the
        // SSA forms (`Ssa`/`SsaList`) map to a Cranelift value-location; the window/fixed *memory*
        // forms are left to Stage 3c (a DWARF memory expression, not a value label). `var_meta[label]`
        // names the variable a label belongs to. Empty unless the module carries `-g` vars.
        let mut var_meta: Vec<(u32, String, Option<u32>)> = Vec::new();
        let mut var_label_map: VarLabelMap = std::collections::HashMap::new();
        if let Some(di) = m.debug_info.as_ref() {
            for v in &di.vars {
                let points: Vec<(u32, u32)> = match &v.loc {
                    // A function-wide SSA index ≈ the block-0 local index (parameters, single-block
                    // promoted scalars — the cases where the two numberings coincide).
                    svm_ir::VarLoc::Ssa { value } => vec![(0, *value)],
                    svm_ir::VarLoc::SsaList(locs) => {
                        locs.iter().map(|l| (l.block, l.value)).collect()
                    }
                    // `Window`/`WindowVia`/`Fixed` are memory locations — no value label here.
                    _ => continue,
                };
                let label = var_meta.len() as u32;
                var_meta.push((v.func, v.name.clone(), v.type_id));
                for (block, value) in points {
                    var_label_map
                        .entry((v.func, block))
                        .or_default()
                        .push((value, label));
                }
            }
        }
        let var_labels = (!var_label_map.is_empty()).then_some(&var_label_map);
        let mut raw_var_locs: RawVarLocs = Vec::new();

        // Define each function body. `clear_context` after each define resets the cached
        // CFG/domtree so the next function never compiles against a stale CFG.
        let mut ctx = module.make_context();
        let mut base_bytes = 0usize;
        for (fi, (f, id)) in m.funcs.iter().zip(&ids).enumerate() {
            // Stage 3a: enable Cranelift's value-label tracking so `set_val_label` (in `lower_block`)
            // takes effect and `value_labels_ranges` is populated. Gated on `-g` vars ⇒ no effect on
            // an ordinary build. Must be set on the fresh `ctx.func` before lowering populates it.
            if var_labels.is_some() {
                ctx.func.collect_debug_info();
            }
            build_clif(
                &mut module,
                &ids,
                &m.funcs,
                &distinct,
                cap,
                fiber,
                thread,
                inst,
                setjmp,
                &mut ctx.func,
                f,
                mask,
                cap_mapped,
                sub_base,
                guard_offset_of(win_reserved as u64),
                epoch_addr,
                (table_len as u64) - 1, // the (possibly B2-reserved) table mask, baked per call site
                fi as u32,
                srcloc_map.as_ref(),
                var_labels,
            )?;
            // Optional codegen dump for performance work: `SVM_JIT_DUMP_CLIF=1` prints each function's
            // CLIF and (post-compile) the Cranelift VCODE disassembly to stderr. A pure diagnostic — no
            // effect unless the env var is set; used to compare svm-jit's generated code against other
            // backends (e.g. the Embench cross-engine bench).
            let dump_codegen = std::env::var_os("SVM_JIT_DUMP_CLIF").is_some();
            if dump_codegen {
                eprintln!("=== SVM-JIT CLIF fn{fi} ===\n{}", ctx.func.display());
                ctx.set_disasm(true);
            }
            module
                .define_function(*id, &mut ctx)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            // NOTE (feature `stack-check`, STACK_GUARD.md §2): soundness wants a post-compile bound
            // rejecting any function whose frame exceeds `RED_ZONE` (so a function can't grow the stack
            // past the limit before its callee re-checks). Cranelift 0.132 doesn't expose the final
            // frame size on `CompiledCode` (`FrameLayout`/`total_stacksize` are ABI-internal), so this
            // bound is DEFERRED — see the increment plan. Until then the check assumes frames ≤ RED_ZONE.
            if dump_codegen {
                if let Some(d) = ctx.compiled_code().and_then(|c| c.vcode.as_ref()) {
                    eprintln!("=== SVM-JIT VCODE fn{fi} ===\n{d}");
                }
            }
            base_bytes += ctx.compiled_code().map_or(0, |c| c.code_buffer().len());
            if srcloc_map.is_some() {
                if let Some(cc) = ctx.compiled_code() {
                    let ranges: Vec<(u32, u32, u32)> = cc
                        .buffer
                        .get_srclocs_sorted()
                        .iter()
                        .filter(|s| !s.loc.is_default())
                        .map(|s| (s.start, s.end, s.loc.bits()))
                        .collect();
                    if !ranges.is_empty() {
                        raw_srclocs.push((fi as u32, ranges));
                    }
                }
            }
            // Stage 3a: capture this function's value-label ranges, translating each Cranelift
            // `LabelValueLoc` to DWARF terms now (the ISA is in hand; a `Reg` we cannot map is
            // dropped — that sub-range simply has no location, like an optimized-out gap).
            if var_labels.is_some() {
                if let Some(cc) = ctx.compiled_code() {
                    let isa = module.isa();
                    let mut per_label: Vec<(u32, VarLocRanges)> = Vec::new();
                    for (label, ranges) in &cc.value_labels_ranges {
                        let mut out: VarLocRanges = Vec::new();
                        for r in ranges {
                            if r.end <= r.start {
                                continue;
                            }
                            let loc = match r.loc {
                                LabelValueLoc::Reg(reg) => match isa.map_regalloc_reg_to_dwarf(reg)
                                {
                                    Ok(d) => VarMachineLoc::Reg(d),
                                    Err(_) => continue,
                                },
                                LabelValueLoc::CFAOffset(off) => VarMachineLoc::CfaOffset(off),
                            };
                            out.push((r.start, r.end, loc));
                        }
                        if !out.is_empty() {
                            per_label.push((label.as_u32(), out));
                        }
                    }
                    if !per_label.is_empty() {
                        raw_var_locs.push((fi as u32, per_label));
                    }
                }
            }
            module.clear_context(&mut ctx);
        }

        // The buffer-ABI trampoline for the entry, exported so Rust can call it.
        build_trampoline(&mut module, &mut ctx.func, ids[func as usize], entry);
        let tramp = module
            .declare_function("trampoline", Linkage::Export, &ctx.func.signature)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        module
            .define_function(tramp, &mut ctx)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        module.clear_context(&mut ctx);

        // The generic call-trampoline (one per module; calls any `Tail`-ABI `(sp, arg) -> i64` entry from
        // Rust). Needed by both the fiber runtime (`cont.*`) and the thread scheduler (vCPU entries).
        #[cfg(fiber_rt)]
        let fiber_tramp = if uses_fibers || uses_threads {
            build_fiber_call_trampoline(&mut module, &mut ctx.func);
            let id = module
                .declare_function("fiber_call_tramp", Linkage::Export, &ctx.func.signature)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            module
                .define_function(id, &mut ctx)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            module.clear_context(&mut ctx);
            Some(id)
        } else {
            None
        };

        module
            .finalize_definitions()
            .map_err(|e| JitError::Backend(e.to_string()))?;

        // W5 JIT/DWARF Stage 1: now that code is finalized, resolve each captured `MachSrcLoc` range
        // (relative to its function's start) to an absolute machine address, building the sorted
        // `machine-pc → source` map `symbolize` consults. Empty without `-g`.
        let (src_ranges, src_files, func_names) = match m.debug_info.as_ref() {
            Some(di) => {
                let mut ranges = Vec::new();
                for (fi, rs) in &raw_srclocs {
                    let base = module.get_finalized_function(ids[*fi as usize]) as u64;
                    for &(start, end, loc) in rs {
                        if let Some(l) = di.locs.get(loc as usize) {
                            ranges.push(SrcRange {
                                lo: base + start as u64,
                                hi: base + end as u64,
                                func: l.func,
                                file: l.file,
                                line: l.line,
                                col: l.col,
                            });
                        }
                    }
                }
                ranges.sort_by_key(|r| r.lo);
                // §6 function names (`func → name`), for `symbolize`, the DWARF `DW_AT_name`, and kill
                // messages — `compute` instead of `fn3`.
                let names = di
                    .func_names
                    .iter()
                    .map(|f| (f.func, f.name.clone()))
                    .collect();
                (ranges, di.files.clone(), names)
            }
            None => (Vec::new(), Vec::new(), std::collections::HashMap::new()),
        };

        // W5 JIT/DWARF Stage 3a: resolve the captured value-label ranges (relative offsets) to
        // absolute machine pcs and group them per source variable. A variable whose label produced
        // no range (the optimizer dropped its value everywhere) still gets a `VarMachineInfo` with
        // empty `ranges` — Stage 3c emits that as a `<optimized out>` location.
        let var_locs: Vec<VarMachineInfo> = if var_meta.is_empty() {
            Vec::new()
        } else {
            let mut by_label: std::collections::HashMap<u32, Vec<VarRange>> =
                std::collections::HashMap::new();
            for (fi, per_label) in &raw_var_locs {
                let base = module.get_finalized_function(ids[*fi as usize]) as u64;
                for (label, ranges) in per_label {
                    let entry = by_label.entry(*label).or_default();
                    for &(s, e, loc) in ranges {
                        entry.push(VarRange {
                            lo: base + s as u64,
                            hi: base + e as u64,
                            loc,
                        });
                    }
                }
            }
            var_meta
                .iter()
                .enumerate()
                .map(|(label, (func, name, type_id))| {
                    let mut ranges = by_label.remove(&(label as u32)).unwrap_or_default();
                    ranges.sort_by_key(|r| r.lo);
                    VarMachineInfo {
                        func: *func,
                        name: name.clone(),
                        type_id: *type_id,
                        ranges,
                    }
                })
                .collect()
        };

        // Now that code is finalized, hand the root fiber runtime its call-trampoline address (and keep it
        // to seed the thread `Domain`'s `Env`, which spawned vCPUs use to call their entry).
        #[cfg(fiber_rt)]
        let mut call_tramp: Option<fiber_rt::FiberCallTramp> = None;
        #[cfg(fiber_rt)]
        if let Some(id) = fiber_tramp {
            let addr = module.get_finalized_function(id);
            // SAFETY: `addr` is the finalized `fiber_call_tramp` with exactly the `FiberCallTramp` ABI.
            let t: fiber_rt::FiberCallTramp = unsafe { std::mem::transmute(addr) };
            if let Some(rt) = &mut fiber_rt {
                rt.set_call_tramp(t);
            }
            call_tramp = Some(t);
        }

        // Build the function table (§3c) now that code addresses are known: power-of-two
        // padded (to `table_len`, possibly B2-reserved beyond the module), AoS, host-owned.
        // `call_indirect` masks the guest index into this; padding/reserved slots trap until
        // `install` fills them.
        let fn_table: Box<[FnEntry]> = (0..table_len)
            .map(|slot| match m.funcs.get(slot) {
                Some(f) => FnEntry::new(
                    type_id_of(
                        &distinct,
                        &FuncType {
                            params: f.params.clone(),
                            results: f.results.clone(),
                        },
                    ),
                    module.get_finalized_function(ids[slot]) as u64,
                ),
                None => FnEntry::padding(),
            })
            .collect();

        let tramp_code = module.get_finalized_function(tramp);
        #[cfg(not(fiber_rt))]
        let _ = &quota;
        Ok(CompiledModule {
            fn_table,
            tramp_code,
            n_params: entry.params.len(),
            n_results: entry.results.len(),
            n_real_funcs: m.funcs.len(),
            distinct,
            cap,
            fiber,
            thread,
            inst,
            setjmp,
            mask,
            cap_mapped,
            sub_base,
            epoch_addr,
            fn_table_mask: (table_len as u64) - 1,
            next_extra: 0,
            extra_bytes: 0,
            base_bytes,
            live_fault_range: None,
            last_trap_backtrace: Vec::new(),
            last_trap_fiber: None,
            win_mapped,
            win_reserved,
            win_size,
            mem_size_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone(),
            restore_prots: Vec::new(),
            durable: false,
            concurrent_durable: false,
            frozen_seed: Vec::new(),
            frozen_out: Vec::new(),
            frozen_vcpus_out: Vec::new(),
            frozen_nested_out: Vec::new(),
            frozen_root_sp_out: 0,
            frozen_vcpu_seed: Vec::new(),
            frozen_nested_seed: Vec::new(),
            thaw_root_sp: DURABLE_SHADOW_BASE + 8, // §12.8 4A.5: empty root extent = frame base (past the SP word)
            freeze_ctl: None,
            #[cfg(fiber_rt)]
            fiber_rt,
            #[cfg(fiber_rt)]
            domain,
            #[cfg(fiber_rt)]
            _nursery: nursery,
            #[cfg(setjmp_rt)]
            _setjmp_rt: setjmp_runtime,
            #[cfg(fiber_rt)]
            call_tramp,
            #[cfg(fiber_rt)]
            fiber_cfg: if uses_fibers {
                Some((fiber_type_id, fiber_mask))
            } else {
                None
            },
            #[cfg(fiber_rt)]
            fiber_table,
            src_ranges,
            src_files,
            func_names,
            var_locs,
            debug_types: m
                .debug_info
                .as_ref()
                .map(|di| di.types.clone())
                .unwrap_or_default(),
            module: OwnedJit::new(module),
        })
    }

    /// Run the compiled entry over a **fresh guest window** on slot-encoded `args` (the run half
    /// of the old one-shot `compile_and_run*`): allocate + seed the window (init bytes, data
    /// segments, RO protection), seed the per-run runtime env, execute under the §5
    /// detect-and-kill guard, snapshot, and tear the window down. The executable code and the
    /// runtimes stay alive in `self`, so `run` can be called again (and
    /// [`Self::define_extra`]-d code stays valid across runs).
    pub fn run(
        &mut self,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
        async_hooks: Option<&dyn AsyncHostHooks>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        let (code, n_params, n_results) = (self.tramp_code, self.n_params, self.n_results);
        // SAFETY: `self` is a unique borrow for the whole call and this path hands no pointer
        // to any handler — the re-entrant powerbox path goes through `run_raw` instead.
        unsafe {
            Self::run_code_raw(
                self,
                code,
                n_params,
                n_results,
                args,
                init_mem,
                snapshot_cap,
                async_hooks,
            )
        }
    }

    /// [`Self::run`] through a **caller-managed raw pointer** — the entry the Phase-2 `Jit`
    /// capability path uses, because its handlers re-enter this module mid-run: the host gives
    /// the `Jit` binding a copy of `this`, calls `run_raw(this, …)`, and while the guest is
    /// suspended inside a synchronous `cap.call` the handler may call
    /// [`Self::define_extra`] / [`Self::invoke_extra`] through its copy. `run_raw` keeps no
    /// Rust reference into `*this` alive across the guarded call (see `run_code_raw`), so the
    /// handler's transient `&mut *this` aliases nothing.
    ///
    /// # Safety
    /// - `this` must point at a live `CompiledModule`, not concurrently accessed by any other
    ///   thread, and **the same pointer value** must be the one handlers use (don't re-derive a
    ///   fresh `&mut` elsewhere while the run is in flight).
    /// - Handlers may call `define_extra` / `invoke_extra` through `this` only while the guest
    ///   is suspended in a synchronous `cap.call` on this thread, and must not call `run` /
    ///   `run_raw` re-entrantly.
    pub unsafe fn run_raw(
        this: *mut CompiledModule,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
        async_hooks: Option<&dyn AsyncHostHooks>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        let (code, n_params, n_results) = {
            let t = &*this;
            (t.tramp_code, t.n_params, t.n_results)
        };
        Self::run_code_raw(
            this,
            code,
            n_params,
            n_results,
            args,
            init_mem,
            snapshot_cap,
            async_hooks,
        )
    }

    /// Run an **incrementally defined** function (a trampoline pointer returned by
    /// [`Self::define_extra`]) over a fresh guest window, exactly like [`Self::run`] runs the
    /// entry. This is the test/demo surface; the Phase-2 `Jit` capability instead uses
    /// [`Self::invoke_extra`] over the *live* window of an in-flight run.
    ///
    /// # Safety
    /// `code` must be a trampoline pointer returned by `define_extra` **on this module**, and
    /// `n_params`/`n_results` must match the parameter/result counts of the function it wraps
    /// (the trampoline reads exactly `n_params` arg slots and writes exactly `n_results` result
    /// slots).
    pub unsafe fn run_extra(
        &mut self,
        code: *const u8,
        n_params: usize,
        n_results: usize,
        args: &[i64],
        init_mem: Option<&[u8]>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        Self::run_code_raw(self, code, n_params, n_results, args, init_mem, None, None)
    }

    /// Invoke an extra trampoline **over the live window of an in-flight run** — the engine of
    /// the `Jit` capability's `invoke` op. Called from inside a `cap.call` handler while the
    /// guest is suspended; `mem_base`/`trap_out` are the values the cap thunk received (the
    /// run's window base and trap cell), so the invoked code reads/writes the guest's own
    /// memory in place and a trap in it propagates exactly like a guest trap.
    ///
    /// Runs under a **nested** detect-and-kill recovery (`run_guarded_range` is re-entrant —
    /// the same §14 child-fault pattern as `compile_child_and_run`): a memory fault in the
    /// invoked code is caught *here*, written to `trap_out` as `MemoryFault`, and this returns
    /// normally — the guest's `cap.call` trap-propagation check then unwinds the domain. Traps
    /// in invoked code are **terminal for the domain** (DESIGN.md §22); a guest wanting
    /// trap isolation uses the `Instantiator`, not `Jit`.
    ///
    /// # Safety
    /// - `this` must be the pointer an in-flight [`Self::run_raw`] on **this thread** is
    ///   executing on (the guest suspended in its synchronous `cap.call`), and the caller must
    ///   hold no Rust reference into `*this` across this call.
    /// - `code` must be a trampoline returned by `define_extra` on this module; `args` must
    ///   cover its param count and `results` its result count.
    /// - `mem_base` and `trap_out` must be the live run's window base and trap cell.
    pub unsafe fn invoke_extra(
        this: *mut CompiledModule,
        code: *const u8,
        args: &[i64],
        results: &mut [i64],
        mem_base: *mut u8,
        trap_out: *mut i64,
    ) -> Result<(), JitError> {
        let (fn_table_ptr, live) = {
            let t = &*this;
            (
                t.fn_table.as_ptr() as *const core::ffi::c_void,
                t.live_fault_range,
            )
        };
        let (lo, hi) = live.ok_or(JitError::Unsupported(
            "invoke_extra outside an in-flight run",
        ))?;
        let faulted = mem::run_guarded_range(
            code,
            args.as_ptr(),
            results.as_mut_ptr(),
            mem_base,
            fn_table_ptr,
            trap_out,
            lo,
            hi,
        );
        if faulted {
            // Detect-and-kill (§5), reported the same way the outer run reports it; the
            // guest's cap.call propagation check sees the cell and unwinds the domain.
            *trap_out = mem::FAULT_TRAP;
        }
        Ok(())
    }

    /// The shared run body: window setup → guarded call → snapshot → teardown. `code` is a
    /// buffer-ABI trampoline owned by the module (the entry's, or an extra function's).
    ///
    /// Structured for **mid-run re-entry**: every reference into `*this` is derived
    /// transiently and dropped before the guarded call (raw pointers extracted up front), so a
    /// `cap.call` handler re-entering through its own copy of `this` while the guest is
    /// suspended aliases no live Rust reference. The fields a re-entrant `define_extra`
    /// mutates (`module`, `distinct`, `next_extra`) are disjoint from everything the in-flight
    /// call reads through raw pointers (`fn_table` is boxed, never grown or moved; the window
    /// is a local).
    ///
    /// # Safety
    /// As [`Self::run_raw`]; additionally `code`/`n_params`/`n_results` must describe a
    /// trampoline owned by this module.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_code_raw(
        this: *mut CompiledModule,
        code: *const u8,
        n_params: usize,
        n_results: usize,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
        async_hooks: Option<&dyn AsyncHostHooks>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        // The trampoline reads exactly `n_params` arg slots; a shorter buffer would be an
        // out-of-bounds read from safe code. (The one-shot wrappers always pass exact-length
        // args; this check makes the now-public entry sound rather than contractual.)
        if args.len() < n_params {
            return Err(JitError::Malformed);
        }
        // §4 (DURABILITY.md): a durable run's §14 nursery must know it — its `instantiate` /
        // `coro_spawn` thunks fail closed there (this JIT child runner cannot yet run a durable
        // child; the interpreter is the reference for durable nesting). Set here, the common
        // bottom of every run entry, because the durable flag is applied by the entry wrappers
        // *after* compile (where the nursery is built).
        {
            let t = &*this;
            if let Some(n) = &t._nursery {
                n.set_durable(t.durable);
            }
        }
        // ---- Setup: references into `*this` live only inside this block. ----
        // Allocate the guest window for this run: `mapped` backed RW bytes inside the reserved
        // virtual range planned at compile time (§4); zero-sized when the module has no memory.
        let (mut window, win_size, mask, fn_table_ptr) = {
            let t = &mut *this;
            let mut window = mem::GuestWindow::new(t.win_mapped, t.win_reserved);
            let win_size = t.win_size;
            // Escape-oracle: seed the window's low bytes so a divergent read/store is observable.
            if let Some(init) = init_mem {
                let n = init.len().min(win_size);
                window.rw_mut()[..n].copy_from_slice(&init[..n]);
            }
            // Initialized data segments (§3a / D40): copy each segment's bytes into the window, then map
            // the `readonly` ones RO (so a guest write to const data faults into the guard, §4/§5). The
            // verifier already bounds every segment to `[0, size)`. Segment offsets are child-relative, so
            // a §14 sub-window shifts them by `sub_base` into the parent backing. Done while fully RW.
            if let Some(size_log2) = t.mem_size_log2 {
                let size = 1u64 << size_log2;
                let rw = window.rw_mut();
                for d in &t.data {
                    let lo = t.sub_base + d.offset.min(size);
                    let hi = t.sub_base + (d.offset + d.bytes.len() as u64).min(size);
                    let (start, end) = (lo as usize, hi as usize);
                    rw[start..end].copy_from_slice(&d.bytes[..end - start]);
                }
                // Map the `readonly` segments RO (so a guest write to const data faults into the guard,
                // §4/§5). Clamp the range to `[0, size)` exactly as the copy loop above: the verifier
                // already bounds every data segment into the window, so this is defensive consistency —
                // an out-of-window segment must never `mprotect` past the backed region.
                for d in &t.data {
                    if !d.readonly {
                        continue;
                    }
                    let lo = d.offset.min(size);
                    let hi = d.offset.saturating_add(d.bytes.len() as u64).min(size);
                    if hi > lo {
                        window.protect_ro(t.sub_base + lo, hi - lo);
                    }
                }
            }
            // Durable restore (DURABILITY.md §12.3): re-establish captured per-page protections
            // on the freshly-seeded window so a thawed guest faults on an `Ro`/`Unmapped` page
            // exactly as the frozen one would — matching `svm-interp`'s `apply_prots`. Applied
            // after the init copy + data segments; `Rw` and tail pages keep the default.
            for (i, &p) in t.restore_prots.iter().enumerate() {
                let off = (i * DURABLE_SNAPSHOT_PAGE) as u64;
                if off >= t.win_mapped as u64 {
                    break;
                }
                match p {
                    WindowProt::Ro => {
                        window.protect_ro(t.sub_base + off, DURABLE_SNAPSHOT_PAGE as u64)
                    }
                    WindowProt::Unmapped => {
                        window.protect_none(t.sub_base + off, DURABLE_SNAPSHOT_PAGE as u64)
                    }
                    WindowProt::Rw => {}
                }
            }
            let fn_table_ptr = t.fn_table.as_ptr() as *const core::ffi::c_void;
            (window, win_size, t.mask, fn_table_ptr)
        };

        let mem_base = window.base();
        let mut results = vec![0i64; n_results];
        // Trap cell: 0 = ok; low 32 bits = a TrapKind / EXIT_CODE; high 32 bits = the exit
        // code for an Exit. A trapping path (or the cap thunk) writes it. It is **shared across vCPU
        // threads** (every spawned vCPU gets its address via `set_env`), so the Rust accesses are atomic
        // (audit #2): the JIT writes it via an aligned `i64` store in emitted code (hardware-atomic,
        // foreign to Rust's model); concurrent Rust writers (dying vCPUs) and this reader must not race.
        let trap_cell = AtomicI64::new(0);

        // §12: the root vCPU (`main`) runs on this thread under the §5 detect-and-kill guard; any spawned
        // vCPUs run on their own OS threads via the baked `Domain`. Seed the `Domain`'s per-run `Env` now
        // that the window / fn-table / trap-cell / call-trampoline are all known.
        #[cfg(fiber_rt)]
        {
            let t = &*this;
            if let Some(d) = &t.domain {
                d.set_env(
                    mem_base as u64,
                    fn_table_ptr as u64,
                    trap_cell.as_ptr(),
                    t.call_tramp
                        .expect("call-trampoline set for a threaded module"),
                    window.fault_range(),
                    t.fiber_cfg,
                    t.fiber_table.clone(), // the domain-shared table spawned vCPUs build over
                    t.epoch_addr as usize, // §5 kill-path: so parked vCPUs (futex/join) observe the interrupt
                    t.durable, // slice 3.3: run spawned children inline (single-worker) under a freeze/thaw
                );
            }
            // §9/§12 async ring: publish this run's futex-`notify` into the embedder's `Host` so an offload
            // worker can wake a vCPU parked in `submit_async` on a completion counter (the futex `phys` is the
            // parking key). Needs the thread `Domain` (a module that parks on a counter uses `atomic.wait`, so
            // `uses_threads` holds). With no `Domain`/hooks, `submit_async` stays an inert `-EINVAL`.
            if let (Some(hooks), Some(d)) = (async_hooks, &t.domain) {
                // The `Domain` pointer as a `usize` so the hook closure is `Send + Sync` (a raw pointer is not,
                // and Rust-2021 disjoint capture would otherwise grab the bare pointer field).
                let dom_addr = (&**d as *const os_thread_rt::Domain) as usize;
                hooks.install_notify(std::sync::Arc::new(move |key: u64, count: u32| {
                    let n = count.min(i32::MAX as u32) as i32;
                    // SAFETY: the `Domain` outlives the run; the hook is dropped by `hooks.finish()` after
                    // `join_all`, before the `Domain` is freed, so the pointer is valid whenever a worker
                    // calls this. `thread_notify` is sound from any thread (it locks the domain futex), like a
                    // guest `atomic.notify`.
                    unsafe {
                        os_thread_rt::thread_notify(dom_addr as *const os_thread_rt::Domain, key, n)
                    };
                }));
            }
        }
        #[cfg(not(fiber_rt))]
        let _ = &async_hooks;

        // Set if a durable thaw re-seed (below) hit a control-stack alloc failure (I1): the trap cell
        // already carries the `FiberFault`, so we skip the root re-entry and report it post-run.
        // (Only the `fiber_rt` build re-seeds, so it's never mutated otherwise.)
        #[cfg_attr(not(fiber_rt), allow(unused_mut))]
        let mut seed_faulted = false;

        // Publish the root fiber runtime (when the module uses `cont.*`) so its thunks find it via the
        // thread-local for the duration of the entry; spawned vCPUs publish their own.
        #[cfg(fiber_rt)]
        let prev_rt = {
            // The OS-thread stack pointer in *this* (`run_inner`) frame — above the guarded guest
            // call below and every guest root frame it pushes — is the high bound for `gc.roots`'
            // scan of the root computation's frames. Captured via a local's address here (not a
            // sub-call) so it provably dominates the guest's frames.
            let entry_probe = 0u8;
            let entry_sp = std::hint::black_box(&entry_probe as *const u8 as usize);
            let t = &mut *this;
            let durable = t.durable;
            // Durable **thaw** (slice 3.3.3): re-create the frozen fibers in the run-shared table
            // before the root re-enters under REWINDING, so a thaw `cont.resume` resolves + re-enters
            // them. Done before `set_current` (and the run), while the window/table/trap cell are set.
            let seed = std::mem::take(&mut t.frozen_seed);
            t.fiber_rt.as_mut().map(|rt| {
                rt.set_root_entry_sp(entry_sp);
                // Arm the durable fiber-switch swap for the root vCPU (DURABILITY.md §12.8): the
                // window base is known now. (Spawned vCPUs are multi-vCPU durability, Phase 3.2.)
                rt.set_durable_env(mem_base as u64, durable);
                if durable && !seed.is_empty() {
                    // A thaw re-seed that the OS refuses (I1) writes a `FiberFault` to the trap cell
                    // and returns false; the post-run trap read below reports it. We still publish
                    // the runtime and fall through — the guest re-entry simply won't resolve the
                    // missing fibers — rather than abort the host.
                    seed_faulted = !fiber_rt::seed_frozen_fibers(
                        &mut **rt as *mut fiber_rt::FiberRuntime,
                        &seed,
                        mem_base as u64,
                        fn_table_ptr as u64,
                        trap_cell.as_ptr() as u64,
                    );
                }
                fiber_rt::set_current(&mut **rt as *mut fiber_rt::FiberRuntime)
            })
        };

        // §12.8 concurrent-thaw stage 2: the parked-vCPU thaw no longer needs a `Domain.thawing` flag —
        // the thaw re-spawns frozen vCPUs as concurrent OS threads (below), so a re-issued `atomic.wait`
        // parks on the real futex and a sibling's re-issued `notify` wakes it; a wait with no possible
        // notifier left fails closed via `futex_wait`'s shared deadlock detection (no thaw-specific path).

        // Durable **multi-vCPU thaw** (slice 3.3, thaw side): re-attach + run the spawned children a
        // freeze flattened, *before* the root re-enters — the JIT's single-worker thaw (the root's
        // rewind skips its prologue `thread.spawn`, so the runtime reconstructs the children). Each
        // child rewinds from its restored extent, runs forward to completion, and publishes its result
        // so the root's re-executed `thread.join` resolves; the root's active shadow-SP is then set to
        // its restored extent. Done after the fiber seed (the table is live) and `set_env`.
        #[cfg(fiber_rt)]
        if (*this).durable && !(*this).frozen_vcpu_seed.is_empty() {
            let seed = std::mem::take(&mut (*this).frozen_vcpu_seed);
            let root_sp = (*this).thaw_root_sp;
            if let Some(d) = &(*this).domain {
                d.thaw_reattach_and_run(&seed, root_sp);
            }
        }

        // §4 nested thaw (JIT parity): re-attach + **rewind** the frozen §14 children before the parent
        // re-enters, so its re-executed `join` resolves to each child's (rewound) result. Unlike a
        // `thread.spawn` child (a peer vCPU, above), a §14 child is a nested *domain* re-run over its
        // carve in thaw mode (rewind from the carve's frozen continuation, not a fresh start); the
        // result is published at the child's join slot for the parent's `join`. Depth-2: re-attach only
        // the **root's direct children** (`parent_task == 0`) here — each re-runs in thaw mode, and
        // `compile_child_and_run` recursively re-attaches *its* grandchildren before it runs. A **shared**
        // counter assigns task ids in DFS re-attach order, reproducing the freeze-time ids so a
        // grandchild's `parent_task` resolves to its re-attached parent-child.
        #[cfg(fiber_rt)]
        if (*this).durable && !(*this).frozen_nested_seed.is_empty() {
            let seed = std::mem::take(&mut (*this).frozen_nested_seed);
            if let Some(n) = &(*this)._nursery {
                let funcs = n.funcs();
                let epoch = n.epoch_addr();
                let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1));
                let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new())); // thaw captures none
                let mut roots: Vec<&FrozenNested> =
                    seed.iter().filter(|s| s.parent_task == 0).collect();
                roots.sort_by_key(|s| s.slot);
                for rec in roots {
                    let my_task = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let entry = rec.entry as FuncIdx;
                    let nargs = funcs
                        .get(entry as usize)
                        .map(|f| f.params.len())
                        .unwrap_or(0);
                    let args = vec![0i64; nargs];
                    // SAFETY: the carve `[carve_off, carve_off + 2^size_log2)` is committed in the
                    // restored window at `mem_base` (it rode the artifact); `compile_child_and_run`
                    // copies it into the child's own guarded window, rewinds it, and copies it back.
                    match compile_child_and_run(
                        &funcs,
                        entry,
                        rec.carve_off,
                        rec.size_log2,
                        mem_base,
                        &args,
                        epoch,
                        true, // durable
                        true, // thaw: rewind from the carve's frozen continuation
                        my_task,
                        std::sync::Arc::clone(&sink),
                        std::sync::Arc::clone(&counter),
                        &seed, // the full subtree residue — each level re-attaches its own children
                    ) {
                        Ok((result, trap, _)) => n.seed_child_result(rec.slot, result, trap),
                        Err(_) => n.seed_child_result(rec.slot, 0, TrapKind::CapFault as i64),
                    }
                }
            }
        }

        // Publish the live window's fault range so a mid-run `invoke_extra` (from a cap.call
        // handler) can arm its nested recovery against this run's window.
        (*this).live_fault_range = Some(window.fault_range());

        // ---- The guarded call: NO Rust reference into `*this` is live here, so a handler's
        // ---- transient `&mut *this` (define_extra / invoke_extra mid-run) aliases nothing.
        // SAFETY: `code` is a finalized buffer-ABI trampoline honouring the `Entry` ABI. It reads the
        // arg slots, writes the result slots, accesses only the guarded window (any escape faults into
        // the guard page), reads `fn_table`, and writes `trap_cell`. All buffers outlive the call;
        // the module owns the executable pages until `*this` drops (after every spawned vCPU is
        // joined below).
        // §12 seed the root vCPU's TLS register to 0 (its dense id), resetting any value a reused
        // worker thread carries from a prior run before guest code can `vcpu.tls.get` it.
        vcpu_tls::seed(0);
        // §12.8 4A.5: seed the durable shadow-base register to the root's region (context 0 =
        // `DURABLE_SHADOW_BASE`), so the root's instrumented code addresses its own per-context
        // shadow-SP word.
        durable_shadow::seed(DURABLE_SHADOW_BASE);
        // §12.8 4A.5 stage (ii): engage the concurrent durable path before the guarded call (where the
        // root may `thread.spawn` children) so each child reserves its own shadow context.
        #[cfg(fiber_rt)]
        if (*this).concurrent_durable {
            if let Some(d) = &(*this).domain {
                d.engage_concurrent_durable();
            }
        }
        // Phase-4 Slice A (4A.3): publish the live window base for an async controller right before the
        // guarded call (the guest is about to block in its loop), and retire it right after — so a
        // `request_freeze` can only ever store into the window while it is mapped (freed below).
        if let Some(fc) = &(*this).freeze_ctl {
            fc.publish(mem_base as usize);
        }
        let faulted = if seed_faulted {
            // A thaw re-seed already failed and wrote the trap; don't re-enter with missing fibers.
            false
        } else {
            mem::run_guarded(
                &window,
                code,
                args.as_ptr(),
                results.as_mut_ptr(),
                mem_base,
                fn_table_ptr,
                trap_cell.as_ptr(),
            )
        };
        if let Some(fc) = &(*this).freeze_ctl {
            fc.retire();
        }

        // §5 W3 — the **root vCPU's** trap-time backtrace capture (this run thread's thread-local): a
        // memory fault's SIGSEGV/SIGBUS-handler walk (Stage 1) or an explicit trap's helper walk
        // (Stage 2). Taken raw now (and cleared, so a clean run stays empty); symbolized after
        // `join_all` below, where it is reconciled with any **spawned vCPU's** capture (Stage 3 —
        // collected into the domain because a worker's thread-local dies with the worker).
        let root_trap_cap = mem::take_trap_frame();

        // Durable freeze driver (DURABILITY.md §12.8 slice 3.3.2): on a durable **freeze** run
        // (state word UNWINDING) the root has now unwound into context 0's shadow region; flatten
        // every still-parked fiber into its own region before the window is snapshotted, so the
        // artifact captures their continuations. CURRENT_RT is still the root runtime here. A
        // flattening fiber touches only the committed reserve, so it's sound outside the guard.
        // This drives the **root's** own fibers; a spawned child flattens *its* fibers in
        // `run_child_inline` (slice 3.4), merged below. Skipped on a fault or a non-freeze run. The
        // residue (incl. any fiber unwound mid-resume-chain during the root run, slice 3.2) is
        // accumulated in the runtime by each fiber's `Complete` arm; drain it after driving — then the
        // deferred children run (`drive_frozen_spawns`).
        #[cfg(fiber_rt)]
        if (*this).durable && !faulted && fiber_rt::window_is_unwinding(mem_base as u64) {
            if let Some(rt) = (*this).fiber_rt.as_mut() {
                let rt = &mut **rt as *mut fiber_rt::FiberRuntime;
                fiber_rt::freeze_drive(rt, trap_cell.as_ptr() as u64);
                (*this).frozen_out = fiber_rt::take_frozen(rt); // read back by the durable entry
            }
            // Slice 3.3: capture the **root's** flattened extent now — the freeze driver restored the
            // active shadow-SP to context 0's region (root rewinds first on thaw), but the children
            // below overwrite the shared word with their own extents, so the root's must be read here
            // (its implicit residue, reported separately for a thaw to restore).
            // §12.8 4A.5: the root's shadow-SP word lives in its own region (context 0); children no
            // longer share it.
            (*this).frozen_root_sp_out =
                fiber_rt::read_shadow_sp(mem_base as u64, fiber_rt::shadow_region_base(0));
            // Slice 3.3: each `thread.spawn` during the freeze *deferred* its child (recording the
            // request, returning the handle) so the root could unwind first — matching the interp,
            // which enqueues a child and runs it only after the spawning vCPU yields. Now that the
            // root has unwound and its fibers are flattened, run the deferred children **inline**
            // (single-worker) in spawn order: each unwinds into its own top-down context and records a
            // `FrozenVCpu`. This reproduces the interp's dispatch order (root → root's fibers →
            // children), so the side-effect interleaving — and the frozen window — is byte-identical.
            if let Some(d) = &(*this).domain {
                d.drive_frozen_spawns();
                // §12.8 4A.5 stage (ii): drains the **deferred** (single-worker) children now;
                // **concurrent** children record their residue on their own OS threads, drained again
                // after `join_all` below. `extend` (vs. assign) so both contribute. Canonical sort at
                // serialize means the append order can't affect the artifact (§12.6).
                (*this).frozen_vcpus_out.extend(d.take_frozen_vcpus());
                // Slice 3.4: a spawned child that owns fibers flattened them with its own
                // `freeze_drive` (in `run_child_inline`) into the domain accumulator; merge that into
                // the run residue alongside the root's. Sort-by-slot at serialize is canonical, so the
                // append order doesn't affect the artifact.
                (*this).frozen_out.extend(d.take_frozen_fibers());
            }
            // §4 freeze export: the §14 nested children that unwound under the freeze recorded their
            // `FrozenNested` residue into the run's `Nursery`; drain it now that the root has unwound,
            // read back by the durable-nested entry point.
            if let Some(n) = &(*this)._nursery {
                (*this).frozen_nested_out = n.take_frozen_nested();
            }
        }

        // ---- Teardown: transient references again. ----
        (*this).live_fault_range = None;
        #[cfg(fiber_rt)]
        if let Some(p) = prev_rt {
            fiber_rt::set_current(p);
        }
        // S1c: join every async §14 child OS thread before freeing the window — a child copies to/from
        // the parent window, so none may outlive it (mirrors the vCPU `join_all` just below).
        #[cfg(fiber_rt)]
        if let Some(n) = &(*this)._nursery {
            n.join_children();
        }
        // Join every spawned vCPU OS thread before freeing the window — no vCPU may outlive it.
        #[cfg(fiber_rt)]
        if let Some(d) = &(*this).domain {
            d.join_all();
            // §12.8 4A.5 stage (ii): concurrent durable children self-unwound into their own regions
            // and recorded their `FrozenVCpu` residue before their OS threads ended; `join_all` is the
            // coordinator-wait (every child finished). Collect that residue now — after the join, so the
            // snapshot below captures a fully-quiesced window. No-op off the concurrent path.
            if (*this).durable && !faulted && fiber_rt::window_is_unwinding(mem_base as u64) {
                (*this).frozen_vcpus_out.extend(d.take_frozen_vcpus());
                // §12.8 4A.5 follow-up A: carry completed concurrent children's join results, so a
                // `thread.join` of a child that finished before the freeze point resolves on thaw.
                (*this)
                    .frozen_vcpus_out
                    .extend(d.take_completed_children_residue());
                // §12.8 4A.5 follow-up B: a concurrent child that owns fibers flattened them in its own
                // `run_child` `freeze_drive`, recorded during `join_all` — drain after the join.
                (*this).frozen_out.extend(d.take_frozen_fibers());
            }
        }
        // §5 W3 Stage 3 — a trap that originated on a *spawned* vCPU stashed its backtrace capture in
        // that worker's thread-local, which dies with the worker; the worker handed it to the domain
        // before finishing, so collect it now (all vCPUs are joined). Used only when the root vCPU
        // itself didn't trap (`root_trap_cap` below takes precedence — the entry's own trap is the
        // primary one in the common single-vCPU case).
        #[cfg(fiber_rt)]
        let worker_trap_cap = (*this).domain.as_ref().and_then(|d| d.take_trap_capture());
        #[cfg(not(fiber_rt))]
        let worker_trap_cap: Option<(usize, Vec<usize>, i64)> = None;
        // §9/§12 async ring: now that every vCPU is joined, drain the offload pool and drop the futex hook
        // (which holds the `Domain` pointer) before the window is freed below — so no worker
        // can still write the window counter or call into a dead `Domain`.
        #[cfg(fiber_rt)]
        if let Some(hooks) = async_hooks {
            hooks.finish();
        }
        // A caught guard fault is detect-and-kill (§5): report MemoryFault to the host. All vCPUs are
        // joined by now (`join_all` above), so this store no longer races; Relaxed is fine.
        if faulted {
            trap_cell.store(mem::FAULT_TRAP, Ordering::Relaxed);
        }
        // Snapshot the in-window bytes (escape-oracle). The guest may have made pages non-readable
        // via the Memory cap (unmap/protect), so restore RW first — else this read faults outside the
        // guarded call and crashes the host.
        window.restore_rw();
        // `snapshot_cap` (the `_with_host` capture) widens the snapshot past the backed prefix to also
        // cover reserved-tail pages the guest grew/`unmap`-ed (§1a growth path), `commit`-ing them so the
        // read sees zero/their content instead of faulting. `read_low` clamps to the reservation.
        let snap = match snapshot_cap {
            Some(cap) if win_size > 0 => cap.min((mask + 1) as usize).max(win_size),
            _ => win_size,
        };
        let final_mem = if snap > win_size {
            window.read_low(snap)
        } else {
            window.rw_mut()[..win_size].to_vec()
        };
        // The window dies with this run; the code, function table, and runtimes stay alive in
        // `*this` for the next `run` / `define_extra` / drop.
        drop(window);

        // Publish the trap-time backtrace + fiber for `last_trap_backtrace`/`last_trap_fiber`: the root
        // vCPU's own capture if it trapped, else a spawned vCPU's (Stage 3). The fiber handle (§23-D57)
        // rides along, captured at the trap instant. Symbolizing is pure (reads the address map), so it
        // is fine here after teardown. Reset on a clean run (every successful run resets both).
        match root_trap_cap.or(worker_trap_cap) {
            Some((pc, rets, fiber)) => {
                (*this).last_trap_backtrace = (*this).trap_backtrace(pc, &rets);
                (*this).last_trap_fiber = Some(fiber);
            }
            None => {
                (*this).last_trap_backtrace = Vec::new();
                (*this).last_trap_fiber = None;
            }
        };

        // Post-`join_all` read: every vCPU has finished, so this load sees the last store (the join is a
        // synchronization point); Relaxed suffices.
        let cell = trap_cell.load(Ordering::Relaxed);
        let code = cell as u32;
        let outcome = if code == 0 {
            JitOutcome::Returned(results)
        } else if code == EXIT_CODE {
            JitOutcome::Exited((cell >> 32) as i32)
        } else {
            JitOutcome::Trapped(TrapKind::from_code(code).ok_or(JitError::Malformed)?)
        };
        Ok((outcome, final_mem))
    }

    /// Declare + define + finalize **additional functions** into the live module (DESIGN.md §22:
    /// the enabling primitive for the guest-driven `Jit` capability). The slice is a
    /// self-contained unit: its `FuncIdx` space is unit-local, so direct calls resolve within the
    /// unit only (cross-unit calls go through `call_indirect` against the parent table, or the
    /// guest re-emits the callee — DESIGN.md §22 "Recommendation"). Returns one buffer-ABI trampoline
    /// pointer per function, in order; invoke via [`Self::run_extra`] (or, in the capability
    /// layer, directly over a live run's window).
    ///
    /// **The function table is deliberately untouched**: extra functions are thunk-reachable
    /// only, so the table mask baked into every existing `call_indirect` site never changes
    /// (DESIGN.md §22 — zero new escape-relevant dispatch surface). Extra code is lowered
    /// against the *parent's* environment: same confinement mask, same `cap.call` thunk, same
    /// table mask, and the module's shared **append-only type-id registry** (see
    /// [`Self::interned_type_id`]) — the unit's signatures are interned before lowering, so
    /// id-equality coincides with structural equality across every unit this module has
    /// compiled or will compile. A `call_indirect` whose signature no table entry carries
    /// still traps, fail-closed — but a signature first introduced here keeps a stable id, so
    /// a future table install of a structurally equal function can satisfy it.
    ///
    /// Functions using §12 fibers/threads are rejected (`Unsupported`) — the MVP restricts
    /// incremental definition to single-threaded code (DESIGN.md §22 "Concurrency"), and lowering
    /// `cont.*`/`thread.*` here would need per-unit runtime wiring this slice doesn't do.
    pub fn define_extra(&mut self, funcs: &[Func]) -> Result<Vec<DefinedFn>, JitError> {
        if funcs.is_empty() {
            return Ok(Vec::new());
        }
        for f in funcs {
            ensure_supported(f)?;
            if f.uses_concurrency() {
                return Err(JitError::Unsupported(
                    "an incrementally defined function using fibers/threads is not supported yet",
                ));
            }
        }
        // Intern the unit's signatures (its functions' own + its call sites') into the
        // append-only registry BEFORE lowering, so the ids baked into this unit's dispatch
        // checks are real, stable ids — id-equality ≡ structural equality across all units
        // sharing this module, past and future (DESIGN.md §22; see `intern_type`).
        intern_unit_sigs(&mut self.distinct, funcs)?;
        // Declare the unit's functions first so intra-unit direct calls can reference any of them.
        let ids: Vec<FuncId> = funcs
            .iter()
            .map(|f| {
                let name = format!("x{}", self.next_extra);
                self.next_extra += 1;
                let sig = natural_sig(&mut self.module, f);
                self.module
                    .declare_function(&name, Linkage::Local, &sig)
                    .map_err(|e| JitError::Backend(e.to_string()))
            })
            .collect::<Result<_, _>>()?;

        let mut ctx = self.module.make_context();
        for (f, id) in funcs.iter().zip(&ids) {
            build_clif(
                &mut self.module,
                &ids,
                funcs,
                &self.distinct,
                self.cap,
                self.fiber,
                self.thread,
                self.inst,
                self.setjmp,
                &mut ctx.func,
                f,
                self.mask,
                self.cap_mapped,
                self.sub_base,
                guard_offset_of(self.win_reserved as u64),
                self.epoch_addr,
                self.fn_table_mask, // the parent's table mask, NOT derived from this unit's size
                0,
                None, // extra/installed units carry no source-loc map (W5 JIT/DWARF)
                None, // …nor value-label points (Stage 3a)
            )?;
            self.module
                .define_function(*id, &mut ctx)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            // Byte-accurate occupancy: the just-emitted code size, read before `clear_context`.
            self.extra_bytes += ctx.compiled_code().map_or(0, |c| c.code_buffer().len());
            self.module.clear_context(&mut ctx);
        }
        // One buffer-ABI trampoline per function, so the host can invoke any of them (any arity).
        let tramp_ids: Vec<FuncId> = funcs
            .iter()
            .zip(&ids)
            .map(|(f, id)| {
                build_trampoline(&mut self.module, &mut ctx.func, *id, f);
                let name = format!("xt{}", self.next_extra);
                self.next_extra += 1;
                let t = self
                    .module
                    .declare_function(&name, Linkage::Export, &ctx.func.signature)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                self.module
                    .define_function(t, &mut ctx)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                self.extra_bytes += ctx.compiled_code().map_or(0, |c| c.code_buffer().len());
                self.module.clear_context(&mut ctx);
                Ok(t)
            })
            .collect::<Result<_, JitError>>()?;
        // Incremental finalize: mprotects only the newly defined code pages; already-finalized,
        // possibly-running code is untouched (the DESIGN.md §22 Phase-1 W^X spike is the test asserting
        // exactly this).
        self.module
            .finalize_definitions()
            .map_err(|e| JitError::Backend(e.to_string()))?;
        // Per function: the buffer-ABI trampoline (for `invoke` over a window) **and** the
        // natural-ABI entry + interned `type_id` (for B2 `install` into the function table —
        // `call_indirect` calls the natural ABI, not the trampoline).
        Ok(funcs
            .iter()
            .zip(&ids)
            .zip(&tramp_ids)
            .map(|((f, id), t)| DefinedFn {
                tramp: self.module.get_finalized_function(*t),
                code: self.module.get_finalized_function(*id),
                type_id: type_id_of(
                    &self.distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
            })
            .collect())
    }

    /// **Install** an incrementally-defined function into the live `call_indirect` table (DESIGN.md §22
    /// Model B2): write its natural-ABI `code` + interned `type_id` into the next reserved
    /// padding slot, returning that slot index — a funcref the guest (or another unit) can
    /// `call_indirect` at native speed (old→new / new→new). `None` if the table is full (every
    /// reserved slot taken). The base never moves (the table was pre-reserved at compile, the
    /// mask is a baked constant), so this is just a slot write. The write is **release-ordered and
    /// atomic** ([`FnEntry::publish`]), so it is sound against a concurrent `call_indirect` from
    /// another thread — the JIT §6 #2 threaded-install path — not only the single-threaded MVP.
    /// Takes `&self`: the running generated code reads the table through raw pointers, so the host
    /// install must not claim a Rust exclusive borrow over it.
    pub fn install(&self, code: *const u8, type_id: u32) -> Option<u32> {
        let slot = self
            .fn_table
            .iter()
            .position(|e| e.type_id() == PADDING_TYPE_ID)?;
        self.fn_table[slot].publish(type_id, code as u64);
        Some(slot as u32)
    }

    /// **Uninstall** a previously-`install`ed `call_indirect` slot (DESIGN.md §22 reclaim): set
    /// it back to a trapping padding slot so the index is reusable by a later `install` and a
    /// stale `call_indirect` of it fails closed (`IndirectCallType`). Returns `true` on success;
    /// `false` for an out-of-range slot, a real module-function slot (`< n_real_funcs`), or an
    /// already-empty slot — a guest may only reclaim what it installed. (The code memory itself
    /// is not freed — cranelift-jit has no per-function free; this reclaims the *slot*.)
    pub fn uninstall(&self, slot: u32) -> bool {
        let i = slot as usize;
        if i < self.n_real_funcs || i >= self.fn_table.len() {
            return false;
        }
        if self.fn_table[i].type_id() == PADDING_TYPE_ID {
            return false; // already empty
        }
        self.fn_table[i].clear();
        true
    }

    /// **Install at a specific slot** — the compaction counterpart of [`Self::install`] (DESIGN.md §22
    /// code-memory compaction). `install` fills the *next* padding slot, which reproduces a dense
    /// install history but not one with `uninstall` gaps; recompaction must reproduce **exact** slot
    /// indices so a funcref a guest already holds keeps resolving to the same unit across the swap.
    /// Writes the unit's natural-ABI `code` + interned `type_id` into `slot`, returning `true` on
    /// success. Refuses (`false`) an out-of-range slot, a real module-function slot
    /// (`< n_real_funcs`, guarding the original program's funcrefs), or an already-occupied slot
    /// (the target must be padding — a fresh module's reserved slots all are). Same trust class as
    /// `install`: a host-driven slot write into a pre-reserved table whose base never moves.
    pub fn install_at(&self, slot: u32, code: *const u8, type_id: u32) -> bool {
        let i = slot as usize;
        if i < self.n_real_funcs || i >= self.fn_table.len() {
            return false;
        }
        if self.fn_table[i].type_id() != PADDING_TYPE_ID {
            return false; // occupied — install_at never overwrites a live slot
        }
        self.fn_table[i].publish(type_id, code as u64);
        true
    }

    /// The currently-occupied **installable** slots (`≥ n_real_funcs`) of the `call_indirect`
    /// table, as `(slot, code, type_id)`. The reclaim driver (DESIGN.md §22) reads this from the
    /// *old* module to learn which units occupy which slots, joins each `code` back to its owning
    /// unit (via the embedder's per-unit install record), and reproduces the exact slot in the
    /// fresh module with [`Self::install_at`]. Real module-function slots (`< n_real_funcs`) are
    /// excluded — they are reproduced by `compile` itself, not by the driver.
    pub fn installed_slots(&self) -> Vec<(u32, u64, u32)> {
        self.fn_table
            .iter()
            .enumerate()
            .skip(self.n_real_funcs)
            .filter(|(_, e)| e.type_id() != PADDING_TYPE_ID)
            .map(|(i, e)| (i as u32, e.code(), e.type_id()))
            .collect()
    }

    /// The number of extra (`define_extra`) functions+trampolines this module has lowered over its
    /// life — a monotonic proxy for code-arena occupancy (cranelift-jit exposes no byte count, and
    /// has no per-function free, so this only grows). An embedder watches it to decide when to
    /// **compact** (DESIGN.md §22): rebuild the live unit set into a fresh module — whose count restarts
    /// near zero — and drop this one, reclaiming the arena. See `tests/jit_compaction.rs`.
    pub fn extra_fn_count(&self) -> usize {
        self.next_extra
    }

    /// **Byte-accurate** code-arena occupancy: the cumulative machine-code bytes of every
    /// `define_extra`'d function + trampoline this module has lowered (the actual emitted size,
    /// summed at finalize — the dominant term in arena consumption; alignment/rodata padding is
    /// excluded, so this slightly *under*counts the true arena bytes). Monotonic; restarts near zero
    /// in a freshly-compacted module. Prefer this over [`Self::extra_fn_count`] for a watermark when
    /// units vary widely in size (a few large functions vs many tiny ones). See
    /// [`crate::CompiledModule`] / `tests/jit_compaction.rs`.
    pub fn extra_byte_count(&self) -> usize {
        self.extra_bytes
    }

    /// Finalized machine-code bytes of the **base module** — every function body emitted by the
    /// initial [`CompiledModule::compile`], summed from Cranelift's `code_buffer` at define time.
    /// The byte-accurate "how big is the JIT'd code" measure (excludes the tiny buffer-ABI
    /// trampoline and later `define_extra` units — those are [`Self::extra_byte_count`]).
    pub fn code_byte_count(&self) -> usize {
        self.base_bytes
    }

    /// Whether a run is in flight on this module (a guarded call published its window fault range).
    /// Compaction (and any rebuild that drops `self`) is only sound at a **quiescent** point — DESIGN.md §22
    /// §6: "it can only run at a quiescent point — the guest is suspended *inside* the very module
    /// being compacted." An embedder asserts `!is_running()` before swapping a freshly-compacted
    /// module in for this one.
    pub fn is_running(&self) -> bool {
        self.live_fault_range.is_some()
    }

    /// The stable type id `ty` was interned under, or `None` if no unit this module compiled
    /// has mentioned it (as a function signature or a call-site type). Ids are append-only —
    /// once returned, an id never remaps — and id-equality coincides with structural equality
    /// over everything this module compiled (see `intern_type`). This is the lookup a table
    /// `install` operation uses to stamp a slot's `type_id` (DESIGN.md §22).
    pub fn interned_type_id(&self, ty: &FuncType) -> Option<u32> {
        self.distinct.iter().position(|t| t == ty).map(|i| i as u32)
    }
}

/// Compile `func` (and the module's other functions, which `call`/`call_indirect` may reach) as a
/// **top-level guest over its own fresh `2^child_size_log2`-byte window**, seeded from the parent's
/// sub-region `[parent_mem_base + sub_base, … + child_size)` and copied back into it on completion.
/// This is the JIT side of §14 nesting (the `Instantiator`): "nesting cost is paid at setup, not at
/// runtime" — the child is compiled once (steady-state per-access cost is the same single AND+ADD as
/// any guest, the masking already fuzzed by the escape-oracle). Running the child over its **own**
/// guarded window (rather than the live parent window) means its confinement, width-overrun guard
/// (detect-and-kill at *its* guard page), and `map`/`unmap` are the ordinary, fully-fuzzed top-level
/// paths — no new escape-TCB codegen — and the parent sees the child's effect as the copy-back at
/// `instantiate`-completion (the §14 superset, materialized at join for a synchronous child). The
/// child runs under a **nested** detect-and-kill guard (`trap_shim`/VEH save+restore the parent's
/// recovery state), so a child fault is caught at the child and the parent's guard stays intact.
///
/// Returns the child's `(result_slot, trap_cell)` — one `i64` result (the Instantiator child returns
/// one `i64`); the trap cell is `0`, a `TrapKind`, or an `Exit` encoding. The child gets an **empty
/// powerbox** (an inert `cap.call`) for now; a child using §12 fibers/threads is rejected
/// (`Unsupported`) — those need per-child runtimes (a follow-up), and null thunks would be unsound.
///
/// # Safety
/// `parent_mem_base` must point at the caller's live guest window with `[sub_base, sub_base +
/// child_size)` committed (the Instantiator checks `sub_base + child_size ≤ holder size`); it must
/// outlive the call. A guest window must already be installed on this thread (`install_guard`).
#[cfg(fiber_rt)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn compile_child_and_run(
    funcs: &[Func],
    child_entry: FuncIdx,
    sub_base: u64,
    child_size_log2: u8,
    parent_mem_base: *mut u8,
    args: &[i64],
    epoch_addr: usize,
    durable: bool,
    thaw: bool,
    my_task: usize,
    nested_sink: std::sync::Arc<std::sync::Mutex<Vec<FrozenNested>>>,
    task_counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    nested_seeds: &[FrozenNested],
) -> Result<(i64, i64, bool), JitError> {
    let child_size = 1u64 << child_size_log2; // bounded ≤ MAX by compile_child's reject (audit #3)

    // §4 (DURABILITY.md, "JIT parity"): a **durable** child gets an attenuated `Instantiator` powerbox
    // so it can nest a grandchild of its own — the freeze slice's prerequisite (an instrumented child
    // that can be *live* mid-computation is one that reached a `cap.call`). Build a child `Nursery`
    // over the child's own funcs, boxed here so its stable address (baked into the child code below)
    // outlives the run; its `instantiate` re-enters `compile_child_and_run` for the grandchild
    // (recursion), re-checking the same durable + same-module guard. The child's powerbox holds one
    // capability — an `Instantiator` over its **own** window (`child_instantiator_thunk`, ctx = the
    // boxed window size); non-`Instantiator` caps and `AddressSpace` are a follow-up. A non-durable
    // child keeps the pre-existing `InstEnv::null()` (can't nest) — behavior unchanged. `child_win_size`
    // is declared before `child_nursery` so it outlives the nursery that borrows it.
    let child_win_size: Box<u64> = Box::new(child_size);
    let child_uses_instantiator = funcs.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::CapCall { type_id: 6, .. }))
        })
    });
    let child_nursery: Option<Box<instantiator_rt::Nursery>> = if durable && child_uses_instantiator
    {
        let n = Box::new(instantiator_rt::Nursery::new(
            funcs.to_vec().into(),
            instantiator_rt::child_instantiator_thunk,
            &*child_win_size as *const u64 as *mut core::ffi::c_void,
            None, // same-module grandchildren only (separate-module is a later slice)
            epoch_addr,
            my_task, // this child's subtree task id (a grandchild it records gets `parent_task = my_task`)
            std::sync::Arc::clone(&task_counter), // shared counter (subtree-wide instantiate order)
            std::sync::Arc::clone(&nested_sink), // shared sink — descendants' residue coalesces at root
        ));
        n.set_durable(true); // the subtree is durable — the grandchild `instantiate` re-checks §4
        Some(n)
    } else {
        None
    };
    let child_inst = match &child_nursery {
        Some(n) => InstEnv {
            nursery_addr: (&**n as *const instantiator_rt::Nursery) as i64,
            instantiate_thunk: instantiator_rt::instantiate as *const () as i64,
            join_thunk: instantiator_rt::join as *const () as i64,
            coro_spawn_thunk: instantiator_rt::coro_spawn as *const () as i64,
            coro_resume_thunk: instantiator_rt::coro_resume as *const () as i64,
            poll_thunk: instantiator_rt::poll as *const () as i64,
            detach_thunk: instantiator_rt::detach as *const () as i64,
            kill_thunk: instantiator_rt::kill as *const () as i64,
            // A durable nested child never installs grant hooks, and the thunks fail durable spawns
            // closed anyway — wired only so the `InstEnv` is fully populated (ops 8/11 → EINVAL/CapFault).
            instantiate_granted_thunk: instantiator_rt::instantiate_granted as *const () as i64,
            instantiate_named_thunk: instantiator_rt::instantiate_named as *const () as i64,
            instantiate_module_named_thunk: instantiator_rt::instantiate_module_named as *const ()
                as i64,
        },
        None => InstEnv::null(),
    };
    // The synchronous child's non-nesting powerbox is empty (an inert `cap.call` → `CapFault`); its
    // `Instantiator` (if any) is routed by `child_inst` above.
    let child = compile_child(
        funcs,
        child_entry,
        child_size_log2,
        empty_cap_thunk,
        core::ptr::null_mut(),
        epoch_addr, // §5 kill-path: the child polls the parent's interrupt cell
        child_inst,
    )?;
    let n_results = funcs[child_entry as usize].results.len();
    let code = child.code;
    let fn_table_ptr = child.fn_table.as_ptr();

    // The child's own fully-mapped window (+ guard page). Seed it from the parent's sub-region so the
    // child starts from the bytes the parent placed there (the §14 data plane is shared memory).
    let mut child_window = mem::GuestWindow::new(child_size as usize, child_size as usize);
    let child_base = child_window.base();
    {
        // SAFETY: `[parent_mem_base + sub_base, … + child_size)` is committed parent-window memory
        // (the Instantiator bounded `sub_base + child_size ≤ holder size ≤ parent size`).
        let src =
            std::slice::from_raw_parts(parent_mem_base.add(sub_base as usize), child_size as usize);
        child_window.rw_mut().copy_from_slice(src);
    }

    // §4: a durable child runs its (possibly instrumented) funcs in the carve as **context 0** of its
    // own window. Seed the ctx-0 durable control words exactly as the interpreter does at the child's
    // first dispatch (`svm-interp` `durable_store_dstate(0, NORMAL)` + `durable_set_sp`): the global
    // state word (`STATE_OFF` = 0) and the ctx-0 thaw word to `NORMAL`, and the ctx-0 shadow-SP word
    // (at `shadow_region_base(0)` = `DURABLE_SHADOW_BASE`) to the empty frame base `shadow_frame_base(0)`.
    // So an instrumented child's prologue sees `NORMAL` and its shadow stack starts empty at the right
    // offset. A valid durable child's window is ≥ `DURABLE_RESERVE` (64 KiB), so these low offsets fit;
    // the size guard keeps a malformed (too-small) guest-requested carve from panicking the host here —
    // such a child instead traps at runtime when its instrumented code reaches past its window.
    const CTX0_SP_OFF: usize = DURABLE_SHADOW_BASE as usize; // shadow_region_base(0) = 64
    const CTX0_THAW_OFF: usize = DURABLE_SHADOW_BASE as usize + 8; // thaw_state_off(0) = 72
    if durable && (child_size as usize) >= CTX0_THAW_OFF + 4 {
        const CTX0_FRAME_BASE: u64 = DURABLE_SHADOW_BASE + 16; // shadow_frame_base(0) = 80
        const STATE_REWINDING: i32 = 2;
        let w = child_window.rw_mut();
        if thaw {
            // §4 thaw: the carve (from the restored artifact) holds the child's frozen continuation —
            // its spilled shadow stack + the ctx-0 SP word pointing at the top of it. Prepare it to
            // **rewind** exactly as `svm-durable::begin_thaw` does for a context: global state word →
            // `NORMAL`, ctx-0 thaw word → `REWINDING`, and **preserve** the SP word + shadow stack (the
            // continuation the rewind replays). The child then dispatches on the thaw word and reloads.
            w[0..4].copy_from_slice(&0i32.to_le_bytes()); // global state word = NORMAL
            w[CTX0_THAW_OFF..CTX0_THAW_OFF + 4].copy_from_slice(&STATE_REWINDING.to_le_bytes());
        } else {
            // §4 freeze/normal: the child inherits the **parent's** durable phase (the interp seeds
            // `child.dstate = parent.durable_state()`). Under a freeze the parent window is `UNWINDING`,
            // so an instrumented child is born unwinding and unwinds at its first poll; under `NORMAL`
            // it runs to completion. Read the parent window's ctx-0 state word (`STATE_OFF` = 0).
            let parent_phase = {
                let p = std::slice::from_raw_parts(parent_mem_base, 4);
                i32::from_le_bytes([p[0], p[1], p[2], p[3]])
            };
            w[0..4].copy_from_slice(&parent_phase.to_le_bytes()); // global state word = parent's phase
            w[CTX0_THAW_OFF..CTX0_THAW_OFF + 4].copy_from_slice(&0i32.to_le_bytes()); // ctx-0 thaw = NORMAL
            w[CTX0_SP_OFF..CTX0_SP_OFF + 8].copy_from_slice(&CTX0_FRAME_BASE.to_le_bytes());
            // ctx-0 SP
        }
    }

    // §4 depth-2 thaw: before the child runs, recursively re-attach + rewind **its own** grandchildren
    // (residue tagged `parent_task == my_task`) over the **child's** window, and publish each result at
    // its slot in the child's nursery — so when the rewinding child re-executes its `join(grandchild)`
    // it resolves without re-running the grandchild (parents-before-children, one level down). The
    // grandchild's `carve_off` is child-window-relative (its `instantiate` resolved the child's own
    // window to `base 0`), so it re-runs over `child_base`. DFS via the shared `task_counter` reproduces
    // the freeze-time task ids, so a grandchild's `parent_task` resolves to this re-attached child.
    if thaw && durable {
        if let Some(cn) = &child_nursery {
            let mut gkids: Vec<&FrozenNested> = nested_seeds
                .iter()
                .filter(|s| s.parent_task == my_task)
                .collect();
            gkids.sort_by_key(|s| s.slot);
            for rec in gkids {
                let gc_task = task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let gc_entry = rec.entry as FuncIdx;
                let gc_nargs = funcs
                    .get(gc_entry as usize)
                    .map(|f| f.params.len())
                    .unwrap_or(0);
                let gc_args = vec![0i64; gc_nargs];
                let out = compile_child_and_run(
                    funcs,
                    gc_entry,
                    rec.carve_off,
                    rec.size_log2,
                    child_base,
                    &gc_args,
                    epoch_addr,
                    true, // durable
                    true, // thaw
                    gc_task,
                    std::sync::Arc::clone(&nested_sink),
                    std::sync::Arc::clone(&task_counter),
                    nested_seeds,
                );
                match out {
                    Ok((r, t, _)) => cn.seed_child_result(rec.slot, r, t),
                    Err(_) => cn.seed_child_result(rec.slot, 0, TrapKind::CapFault as i64),
                }
            }
        }
    }

    let mut results = vec![0i64; n_results];
    let mut trap_cell: i64 = 0;
    // SAFETY: `code` honours the `Entry` ABI; it accesses only its own window `[child_base, …+size)`
    // (baked masking; a width-overrun hits this window's guard page), reads the child `fn_table`, and
    // writes its result/trap slots. The guard is re-entrant, so a child fault is caught here and the
    // parent's recovery state is restored.
    // §4: while a durable child runs (synchronously, this OS thread), point the per-thread durable
    // shadow-base register at the child's own ctx-0 region and restore the parent's after. Every
    // nesting level is ctx 0 of its *own* window, so the value is always `DURABLE_SHADOW_BASE` — the
    // save/restore is value-neutral here but keeps the register correct if a level ever runs at a
    // non-zero context.
    let saved_shadow = durable.then(|| {
        let s = durable_shadow::get();
        durable_shadow::seed(DURABLE_SHADOW_BASE);
        s
    });
    let faulted = mem::run_guarded(
        &child_window,
        code,
        args.as_ptr(),
        results.as_mut_ptr(),
        child_base,
        fn_table_ptr as *const core::ffi::c_void,
        &mut trap_cell,
    );
    if let Some(s) = saved_shadow {
        durable_shadow::seed(s);
    }
    if faulted {
        trap_cell = mem::FAULT_TRAP;
    }
    // Copy the child's final window back into the parent's sub-region — the parent (the superset) now
    // sees the child's writes (materialized at `instantiate`-completion for a synchronous child). A
    // guest with no Memory cap leaves every page mapped; `restore_rw` is defensive.
    // §4 freeze export: a durable child that left its carve `UNWINDING` unwound mid-run under a freeze
    // (spilled its continuation into the carve + returned a placeholder) instead of completing. Detect
    // it from the carve's state word before the copy-back carries the carve into the parent window.
    // The caller (`instantiate`) turns this into a `FrozenNested` re-attach record.
    let unwound = durable && !faulted && fiber_rt::window_is_unwinding(child_base as u64);
    child_window.restore_rw();
    {
        let dst = std::slice::from_raw_parts_mut(
            parent_mem_base.add(sub_base as usize),
            child_size as usize,
        );
        dst.copy_from_slice(&child_window.rw_mut()[..child_size as usize]);
    }
    drop(child_window);
    drop(child); // frees the child's executable memory now the call has returned
    Ok((results.first().copied().unwrap_or(0), trap_cell, unwound))
}

/// A compiled §14 child: the owning [`JITModule`] (executable memory lives until drop), its
/// power-of-two-padded function table, and the entry's buffer-ABI trampoline. Produced by
/// [`compile_child`]; the synchronous Instantiator child runs it once and drops it, a co-fiber
/// child keeps it alive across suspends (the [`instantiator_rt`] coroutine owns it).
#[cfg(fiber_rt)]
pub(crate) struct ChildCode {
    /// The padded function table `call_indirect` dispatches through; its address is baked into the
    /// running code, so it must not move while the child can run (it is boxed and owned here).
    pub(crate) fn_table: Box<[FnEntry]>,
    /// The entry trampoline (buffer ABI, [`mem::run_guarded`]-compatible).
    pub(crate) code: *const u8,
    /// Owns the executable memory; dropped last, and the drop **releases** the code arena back to
    /// the OS (see [`OwnedJit`] — a bare `JITModule` would leak 256 MiB of reservation per child,
    /// eagerly commit-charged on Windows).
    module: OwnedJit,
}

#[cfg(fiber_rt)]
impl Drop for ChildCode {
    fn drop(&mut self) {
        // This impl exists to document that `code`/`fn_table` die with the struct (no use may
        // outlive it) — that contract is what makes the [`OwnedJit`] free-on-drop sound here.
        let _ = &self.module;
    }
}

// PROCESS.md S1c — a compiled child is shareable across OS threads. `ChildCode` is `!Send`/`!Sync`
// only because of the raw `code: *const u8`; it holds no interior mutability and, once
// `finalize_definitions` has run (before it is ever stored), the code arena and `fn_table` are
// **immutable, read-execute memory** — the entry trampoline reads them, never writes. So handing
// `&ChildCode` (or an `Arc<ChildCode>`) to a spawned child thread and running the same code
// concurrently on N threads is sound: every thread only reads the same finalized bytes and jumps into
// them, and the single `OwnedJit` frees the arena once when the last `Arc` drops (after `join_all`, so
// no thread still runs the code). This is the foundation the S1c OS-thread child executor stands on
// (the per-carve cache below becomes `Arc`-backed to match).
#[cfg(fiber_rt)]
unsafe impl Send for ChildCode {}
#[cfg(fiber_rt)]
unsafe impl Sync for ChildCode {}

// Compile-time proof that the child artifact stays thread-shareable — if a future field reintroduced a
// `!Send`/`!Sync` type (e.g. an `Rc` or `Cell`) without a matching soundness review, this assertion
// would fail to compile, catching the regression at the type level (the S1c executor relies on it).
#[cfg(fiber_rt)]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ChildCode>();
};

/// Compile a §14 child module: every function confined (top-level masking) to a fresh
/// `2^child_size_log2`-byte window, `cap.call`s baked to `cap_thunk`/`cap_ctx`, and the entry
/// wrapped in a buffer-ABI trampoline. "Nesting cost is paid at setup" (§14): this is the setup.
/// A child using §12 fibers/threads is rejected (`Unsupported`) — those need per-child runtimes,
/// and compiling them against null thunks would be unsound.
#[cfg(fiber_rt)]
fn compile_child(
    funcs: &[Func],
    child_entry: FuncIdx,
    child_size_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    epoch_addr: usize,
    inst_env: InstEnv,
) -> Result<ChildCode, JitError> {
    // Audit #3: reject an oversize child window explicitly rather than silently clamping with
    // `.min(MAX_JIT_WINDOW_LOG2)`, so the window built here always equals the size the Instantiator
    // *validated* (which requires `child ≤ parent ≤ 2^MAX`, so this is unreachable in practice — but
    // it keeps the invariant local instead of relying on the cross-module parent-size cap).
    if child_size_log2 > MAX_JIT_WINDOW_LOG2 {
        return Err(JitError::Unsupported(
            "child window exceeds the reference JIT's max",
        ));
    }
    let entry = funcs
        .get(child_entry as usize)
        .ok_or(JitError::Malformed)?
        .clone();
    for f in funcs {
        ensure_supported(f)?;
        // A child using §12 fibers/threads would compile against null fiber/thread thunks (no
        // per-child runtime yet) — reject rather than emit a call through a null pointer.
        if f.uses_concurrency() {
            return Err(JitError::Unsupported(
                "a §14 JIT child using fibers/threads is not supported yet",
            ));
        }
        // Likewise `setjmp`/`longjmp`: the child gets a null `SetjmpEnv` (no per-child `setjmp` table
        // yet), so reject rather than bake a null table address into a `SetJmp` site.
        if f.uses_setjmp() {
            return Err(JitError::Unsupported(
                "a §14 JIT child using setjmp/longjmp is not supported yet",
            ));
        }
    }
    let child_size = 1u64 << child_size_log2; // bounded ≤ MAX by compile_child's reject (audit #3)
    let mask = child_size - 1;

    let mut flags = settings::builder();
    let _ = flags.set("is_pic", "false");
    let _ = flags.set("preserve_frame_pointers", "true");
    let _ = flags.set("opt_level", "speed"); // match the top-level compile (GVN/CSE/const-mat)
                                             // Pin `enable_probestack` OFF, as in the top-level compile — the software stack guard's soundness
                                             // depends on `sub rsp` touching no pages before the entry-block check (see the other flag builder).
    let _ = flags.set("enable_probestack", "false");
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Backend(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Backend(e.to_string()))?;
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    if let Ok(arena) = cranelift_jit::ArenaMemoryProvider::new_with_size(256 << 20) {
        builder.memory_provider(Box::new(arena));
    }
    let mut module = JITModule::new(builder);

    let ids: Vec<_> = funcs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let sig = natural_sig(&mut module, f);
            module
                .declare_function(&format!("f{i}"), Linkage::Local, &sig)
                .map_err(|e| JitError::Backend(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let distinct = distinct_types(funcs);

    let cap = CapEnv {
        thunk_addr: cap_thunk as *const () as i64,
        ctx_addr: cap_ctx as i64,
        fast_resolver: None, // nested child: `cap.call`s go to the coroutine thunk, not a fast path
    };
    let mut ctx = module.make_context();
    for (f, id) in funcs.iter().zip(&ids) {
        build_clif(
            &mut module,
            &ids,
            funcs,
            &distinct,
            cap,
            FiberEnv::null(),
            ThreadEnv::null(),
            inst_env, // §4: a durable child's baked nested-nursery `InstEnv` (else null — no nesting)
            SetjmpEnv::null(), // a child using setjmp is rejected below (no per-child runtime yet)
            &mut ctx.func,
            f,
            mask,
            child_size, // the child is fully mapped (reserved == mapped == size)
            0,          // top-level confinement over the child's own window
            guard_offset_of(child_size), // its own window's trailing guard
            epoch_addr as i64, // §5 kill-path: the child polls the parent's interrupt cell
            (ids.len().next_power_of_two() as u64) - 1, // the child's own table mask
            0,
            None, // nested-child units carry no source-loc map (W5 JIT/DWARF)
            None, // …nor value-label points (Stage 3a)
        )?;
        module
            .define_function(*id, &mut ctx)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        module.clear_context(&mut ctx);
    }
    build_trampoline(
        &mut module,
        &mut ctx.func,
        ids[child_entry as usize],
        &entry,
    );
    let tramp = module
        .declare_function("child_trampoline", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module
        .define_function(tramp, &mut ctx)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Backend(e.to_string()))?;

    let table_len = funcs.len().next_power_of_two();
    let fn_table: Box<[FnEntry]> = (0..table_len)
        .map(|slot| match funcs.get(slot) {
            Some(f) => FnEntry::new(
                type_id_of(
                    &distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
                module.get_finalized_function(ids[slot]) as u64,
            ),
            None => FnEntry::padding(),
        })
        .collect();

    let code = module.get_finalized_function(tramp);
    // PROCESS.md S1: a child module was actually JIT-compiled (as opposed to served from the
    // per-carve cache). Counting successful compiles is what lets a test prove the cache hits.
    CHILD_COMPILES.fetch_add(1, Ordering::Relaxed);
    Ok(ChildCode {
        fn_table,
        code,
        module: OwnedJit::new(module),
    })
}

/// PROCESS.md S1: process-wide count of §14 child modules JIT-compiled. `instantiator_rt`'s
/// per-carve compile cache serves repeat spawns of the same `(module, entry, size)` from one
/// compilation — so a shell spawning the same applet N times compiles it once. Monotone; the
/// public [`child_compiles`] reads it (a real metric, and the cache-hit test's observable).
#[cfg(fiber_rt)]
pub(crate) static CHILD_COMPILES: AtomicU64 = AtomicU64::new(0);

/// PROCESS.md S1: how many §14 child modules this process has JIT-compiled (0 where nesting is
/// unsupported — no child ever compiles). A repeat spawn of a cached `(module, entry, size)` does
/// **not** advance this; the metric is the compile-cache's observable.
pub fn child_compiles() -> u64 {
    #[cfg(fiber_rt)]
    {
        CHILD_COMPILES.load(Ordering::Relaxed)
    }
    #[cfg(not(fiber_rt))]
    {
        0
    }
}

/// PROCESS.md S1: compile a **non-durable** §14 child — the cacheable case. Its powerbox is empty
/// (an inert `cap.call` → `CapFault`) and it has no nesting `InstEnv`, so the compiled code depends
/// only on `(funcs, entry, size_log2, epoch_addr)` — nothing per-spawn — which is what makes it
/// safe to cache and reuse across carves (the base is a runtime arg to [`run_child_code`], not
/// baked). The durable / nesting child keeps the per-call [`compile_child_and_run`] path (its baked
/// per-child nursery makes its code un-shareable).
#[cfg(fiber_rt)]
pub(crate) fn compile_nondurable_child(
    funcs: &[Func],
    child_entry: FuncIdx,
    child_size_log2: u8,
    epoch_addr: usize,
) -> Result<ChildCode, JitError> {
    compile_child(
        funcs,
        child_entry,
        child_size_log2,
        empty_cap_thunk,
        core::ptr::null_mut(),
        epoch_addr,
        InstEnv::null(),
    )
}

/// PROCESS.md S1: run an already-compiled non-durable §14 child confined to the carve
/// `[parent_mem_base + sub_base, … + 2^size_log2)`. Because [`compile_child`] bakes only the size
/// mask and the window **base is a runtime arg** to `run_guarded`, one [`ChildCode`] runs at *any*
/// carve offset — the property the compile cache relies on. Allocates the child's own fresh guarded
/// window, seeds it from the carve (the §14 data plane is shared memory), runs under the re-entrant
/// detect-and-kill guard, and copies the result window back into the carve (the parent is the
/// superset). Non-durable only: no ctx-0 / shadow seeding and no freeze-unwind export (a non-durable
/// run never freezes), so this is the `compile_child_and_run` body minus all its durable branches.
///
/// # Safety
/// `code` is a live compiled child (kept alive by the cache for the call). `[parent_mem_base +
/// sub_base, … + child_size)` is committed parent-window memory (the `Instantiator` bounded the
/// carve to the holder's range). `args` matches the entry's arity.
#[cfg(fiber_rt)]
pub(crate) unsafe fn run_child_code(
    code: &ChildCode,
    sub_base: u64,
    child_size_log2: u8,
    parent_mem_base: *mut u8,
    args: &[i64],
    n_results: usize,
) -> (i64, i64) {
    let child_size = 1u64 << child_size_log2;
    let mut child_window = mem::GuestWindow::new(child_size as usize, child_size as usize);
    let child_base = child_window.base();
    {
        // SAFETY: the carve is committed parent memory (Instantiator-bounded), size = child_size.
        let src =
            std::slice::from_raw_parts(parent_mem_base.add(sub_base as usize), child_size as usize);
        child_window.rw_mut().copy_from_slice(src);
    }
    let mut results = vec![0i64; n_results];
    let mut trap_cell: i64 = 0;
    // SAFETY: `code` honours the `Entry` ABI and accesses only its own window (baked size mask; a
    // width-overrun hits this window's guard page); the guard is re-entrant so a child fault is
    // caught here, not propagated to the parent's frame.
    let faulted = mem::run_guarded(
        &child_window,
        code.code,
        args.as_ptr(),
        results.as_mut_ptr(),
        child_base,
        code.fn_table.as_ptr() as *const core::ffi::c_void,
        &mut trap_cell,
    );
    if faulted {
        trap_cell = mem::FAULT_TRAP;
    }
    child_window.restore_rw();
    {
        // The parent (superset) now sees the child's writes: copy the carve back.
        let dst = std::slice::from_raw_parts_mut(
            parent_mem_base.add(sub_base as usize),
            child_size as usize,
        );
        dst.copy_from_slice(&child_window.rw_mut()[..child_size as usize]);
    }
    (results.first().copied().unwrap_or(0), trap_cell)
}

/// The natural CLIF signature for an IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. Both context pointers are threaded through every call so loads/
/// stores reach the window and `call_indirect` reaches the function table.
fn natural_sig(module: &mut JITModule, f: &Func) -> cranelift_codegen::ir::Signature {
    sig_from(module, &f.params, &f.results)
}

/// Max function results returned **in registers**. Above this the JIT spills results to a
/// caller-provided memory **return-area (sret) pointer** — like wasm engines do for multi-value —
/// so a many-result function compiles **uniformly on every target**. (Cranelift's `Tail` calling
/// convention caps register returns at a per-ABI budget: x86-64 fits 8, aarch64 fewer, so returning
/// the count in registers was the one place a *valid* module compiled on one supported target and was
/// rejected on another.) `4` keeps every real signature — including the §12 `(sp,arg)->i64` fiber/
/// thread entry and the multi-result test cases — on the fast register path, while being safely
/// within the tightest target's budget; only `>4`-result functions take the sret path. The decision
/// is by result **count**, a property of the function *type*, so it is identical at every call site —
/// direct, `call_indirect` (its type id pins the same choice), and tail calls.
const MAX_REG_RESULTS: usize = 4;

/// Whether a function with these results returns via the memory return-area pointer (sret) rather
/// than registers — see [`MAX_REG_RESULTS`].
fn uses_sret(results: &[ValType]) -> bool {
    results.len() > MAX_REG_RESULTS
}

/// The sret return-area uses 8-byte slots (`encode_slot`/`decode_slot`, the buffer-ABI encoding), so
/// a `v128` result cannot be carried through it. A `>4`-result signature containing a `v128` is
/// therefore rejected uniformly (`Unsupported`) rather than miscompiled — an exotic non-case (`v128`
/// buffer slots are already out of MVP scope, §17), and the interpreter still covers it.
fn sret_blocked_by_v128(results: &[ValType]) -> bool {
    uses_sret(results) && results.iter().any(|t| matches!(t, ValType::V128))
}

/// The natural signature for an explicit param/result list (shared by `natural_sig`
/// and the `call_indirect` signature import).
fn sig_from(
    module: &mut JITModule,
    params: &[ValType],
    results: &[ValType],
) -> cranelift_codegen::ir::Signature {
    let mut sig = module.make_signature();
    // The `tail` calling convention so `return_call` (guaranteed tail calls, §3b) is
    // available; a normal `call` from the trampoline works against it too.
    sig.call_conv = cranelift_codegen::isa::CallConv::Tail;
    sig.params.push(AbiParam::new(I64)); // mem_base
    sig.params.push(AbiParam::new(I64)); // fn_table_base
    sig.params.push(AbiParam::new(I64)); // trap_out (host-owned trap cell)
                                         // §2b path B: a 4th context param — the running stack's low bound (`usable_low`), or 0 for the
                                         // root/thread-top (which keeps its OS-guarded stack). Threaded through every call (constant within
                                         // a stack's call tree; set anew at each fiber/root entry), so the prologue check reads a per-vCPU
                                         // limit from a register with no cell and no TLS. Always present (the software stack-overflow guard
                                         // is in the always-on escape-TCB path; see `emit_stack_check`).
    sig.params.push(AbiParam::new(I64)); // stack_limit
    let sret = uses_sret(results);
    if sret {
        // The return-area pointer: the callee writes its results here (8-byte slots) instead of
        // returning them in registers, so the result count is target-independent. Placed right
        // after the context pointers, before the user params — the order every call site mirrors.
        sig.params.push(AbiParam::new(I64));
    }
    for p in params {
        sig.params.push(AbiParam::new(clif_ty(*p)));
    }
    if !sret {
        for r in results {
            sig.returns.push(AbiParam::new(clif_ty(*r)));
        }
    }
    sig
}

/// Reject functions using any op outside the integer slice, so `build_clif` can lower
/// the remainder totally. Keeping the check separate keeps the lowering readable.
fn ensure_supported(f: &Func) -> Result<(), JitError> {
    // The sret return-area (used for `>MAX_REG_RESULTS` results) carries 8-byte slots, so a
    // many-result signature containing a `v128` can't pass through it — reject uniformly on every
    // target (the interpreter still covers it). This function's own results + any indirect
    // call/tail-call target type; direct callees are checked as their own definitions.
    if sret_blocked_by_v128(&f.results) {
        return Err(JitError::Unsupported("v128 in a many-result signature"));
    }
    for blk in &f.blocks {
        if let Terminator::ReturnCallIndirect { ty, .. } = &blk.term {
            if sret_blocked_by_v128(&ty.results) {
                return Err(JitError::Unsupported("v128 in a many-result signature"));
            }
        }
        for inst in &blk.insts {
            if let Inst::CallIndirect { ty, .. } = inst {
                if sret_blocked_by_v128(&ty.results) {
                    return Err(JitError::Unsupported("v128 in a many-result signature"));
                }
            }
        }
    }
    for blk in &f.blocks {
        for inst in &blk.insts {
            match inst {
                Inst::ConstI32(_)
                | Inst::ConstI64(_)
                | Inst::ConstF32(_)
                | Inst::ConstF64(_)
                | Inst::Select { .. }
                | Inst::IntCmp { .. }
                | Inst::Eqz { .. }
                | Inst::FBin { .. }
                | Inst::FUn { .. }
                | Inst::Fma { .. }
                | Inst::FCmp { .. }
                | Inst::FToISat { .. }
                | Inst::FToITrap { .. }
                | Inst::IToFConv { .. }
                | Inst::Cast { .. }
                | Inst::PtrAdd { .. }
                | Inst::PtrCast { .. }
                | Inst::Load { .. }
                | Inst::Store { .. }
                | Inst::MemCopy { .. }
                | Inst::MemMove { .. }
                | Inst::MemFill { .. }
                | Inst::AtomicLoad { .. }
                | Inst::AtomicStore { .. }
                | Inst::AtomicRmw { .. }
                | Inst::AtomicCmpxchg { .. }
                | Inst::AtomicFence { .. }
                | Inst::Call { .. }
                | Inst::CallIndirect { .. }
                | Inst::CapCall { .. }
                | Inst::CallImport { .. }
                | Inst::ImportAttach { .. }
                | Inst::RefFunc { .. }
                | Inst::IntBin { .. }
                | Inst::Convert { .. } => {}
                Inst::IntUn { .. } => {}
                // §17 SIMD (D58): all lowered via Cranelift's native vector ops.
                Inst::ConstV128(_)
                | Inst::V128Load { .. }
                | Inst::V128Store { .. }
                | Inst::Splat { .. }
                | Inst::ExtractLane { .. }
                | Inst::ReplaceLane { .. }
                | Inst::VFloatBin { .. }
                | Inst::VFloatUn { .. }
                | Inst::VBitBin { .. }
                | Inst::VNot { .. }
                | Inst::Bitselect { .. }
                | Inst::Shuffle { .. }
                | Inst::Swizzle { .. }
                | Inst::SimdWidthBytes => {}
                // §7 reflection: lowered to a `cap.call` thunk with the reserved `CAP_SELF_TYPE_ID`,
                // serviced host-side like any cap op — so it matches the interpreter.
                Inst::CapSelfCount
                | Inst::CapSelfAttest
                | Inst::CapSelfGet { .. }
                | Inst::CapSelfResolve { .. }
                | Inst::CapSelfLabel { .. } => {}
                // §12 per-vCPU TLS register: a baked thunk over a thread-local — substrate-independent
                // (works for a plain non-fiber root), so supported on every target.
                Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } => {}
                // §12.8 4A.5 durable-runtime-internal: a baked thunk over a per-OS-thread word (like
                // the TLS register), so supported on every target.
                Inst::DurableShadowBase => {}
                // `i64x2` min/max has no single-instruction lowering on the target ISAs, so Cranelift
                // can't legalize it; bail to `Unsupported` (the interp oracle still covers it, and wasm
                // never emits it — `i64x2` has no min/max op). `i8x16.mul` *is* now lowered (widen →
                // `i16x8` multiply → low-byte pack; see the `VIntBin` lowering), so it stays supported.
                Inst::VIntBin { shape, op, .. }
                    if !matches!(
                        (*shape, *op),
                        (
                            VShape::I64x2,
                            VIntBinOp::MinS
                                | VIntBinOp::MinU
                                | VIntBinOp::MaxS
                                | VIntBinOp::MaxU
                        )
                    ) => {}
                // Lane compares lower to a single Cranelift `icmp`/`fcmp` (legalize on every target).
                Inst::VIntCmp { .. } | Inst::VFloatCmp { .. } => {}
                // Lane shifts lower to vector `ishl`/`ushr`/`sshr`; Cranelift legalizes every shape
                // (incl. `i8x16`, which has no native per-byte shift on x86 — it emits a sequence).
                Inst::VShift { .. } => {}
                // Lane `abs`/`neg` lower to vector `iabs`/`ineg`; Cranelift legalizes every shape.
                Inst::VIntUn { .. } => {}
                // `i8x16.popcnt` lowers to a vector `popcnt` (native `cnt` on aarch64, a byte
                // shuffle sequence on x86 — Cranelift legalizes both).
                Inst::VPopcnt { .. } => {}
                // `avgr_u` (`i8x16`/`i16x8` only, verifier-enforced) → native `avg_round`.
                Inst::VAvgr { .. } => {}
                // `i32x4.dot_i16x8_s` / `i16x8.dot_i8x16_s` → `swiden_low/high` + `imul` +
                // `iadd_pairwise` (all legalize).
                Inst::VDot { .. } | Inst::VDotI8 { .. } => {}
                // Extended multiply → widen low/high both operands + `imul` on the wide shape.
                // `imul` legalizes for every wide shape (incl. `i64x2`, unlike `i8x16.mul`).
                Inst::VExtMul { .. } => {}
                // Extended pairwise add → `swiden/uwiden` low+high + `iadd_pairwise` (all legalize).
                Inst::VExtAddPairwise { .. } => {}
                // Q15 rounding multiply → native `sqmul_round_sat`.
                Inst::VQ15MulrSat { .. } => {}
                // Fused multiply-add (relaxed_madd/nmadd) → vector `fma` (one rounding; the same
                // correctly-rounded result the interp's `mul_add` gives, so the differential holds).
                Inst::VFma { .. } => {}
                // Boolean reductions → a scalar `i32` (`vany_true`/`vall_true`/`vhigh_bits`).
                Inst::VAnyTrue { .. } | Inst::VAllTrue { .. } | Inst::VBitmask { .. } => {}
                // Saturating add/sub (`i8x16`/`i16x8` only, verifier-enforced) lower to native
                // `sadd_sat`/`uadd_sat`/`ssub_sat`/`usub_sat`.
                Inst::VSatBin { .. } => {}
                // Widen lowers to `swiden_low`/`uwiden_low`/`*_high`; narrow to `snarrow`/`unarrow`.
                Inst::VWiden { .. } | Inst::VNarrow { .. } => {}
                // Int↔float / float↔float conversions → Cranelift `fcvt_*` / `fvdemote`/`fvpromote_low`.
                Inst::VConvert { .. } => {}
                // pmin/pmax lower to a single `fcmp` plus `bitselect` (both legalize on every target).
                Inst::VPMinMax { .. } => {}
                // §12 fibers/threads: lowered to host runtime calls, but only where the stack-switch
                // substrate exists (`svm_fiber::supported()` — x86-64 unix). Elsewhere, bail so the
                // differential harness skips rather than miscompiles.
                Inst::ContNew { .. }
                | Inst::ContResume { .. }
                | Inst::Suspend { .. }
                | Inst::ThreadSpawn { .. }
                | Inst::ThreadJoin { .. }
                | Inst::MemoryWait { .. }
                | Inst::MemoryNotify { .. }
                // §GC `gc.roots`: scans the live fiber stacks via the fiber runtime — supported only
                // where the stack-switch substrate exists (else it bails like the other fiber ops and
                // the interpreter covers it).
                | Inst::GcRoots { .. }
                    if cfg!(fiber_rt) => {}
                // `setjmp`/`longjmp` (LLVM.md §"JIT `longjmp`", Option B): libc `_setjmp`/`_longjmp`
                // called inline from JITted code with a host-side `jmp_buf` table. Supported on the
                // `setjmp_rt` targets (unix among `fiber_rt`); elsewhere bail so the interpreters cover
                // it (module-granular fallback).
                Inst::SetJmp { .. } | Inst::LongJmp { .. } if cfg!(setjmp_rt) => {}
                _ => return Err(JitError::Unsupported("instruction")),
            }
        }
        match &blk.term {
            Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::BrTable { .. }
            | Terminator::Return(_)
            | Terminator::ReturnCall { .. }
            | Terminator::ReturnCallIndirect { .. }
            | Terminator::Unreachable => {}
        }
    }
    Ok(())
}

/// The host `cap.call` thunk + ctx addresses, baked into each `cap.call` as constants.
#[derive(Clone, Copy)]
struct CapEnv {
    thunk_addr: i64,
    ctx_addr: i64,
    /// The optional D45 devirtualize-to-direct-call resolver (top-level compile only; `None` for
    /// nested children, whose `cap.call`s go to the coroutine thunk). Invoked at compile time.
    fast_resolver: Option<FastCapResolver>,
}

/// The three `cont.*` thunk addresses, baked into `cont.new`/`cont.resume`/`suspend` sites as
/// constants. All `0` (`null`) when the module uses no fibers or the target has no stack-switch
/// support. The fiber *runtime* itself is found via a thread-local (per vCPU), not baked here.
#[derive(Clone, Copy)]
struct FiberEnv {
    new_thunk: i64,
    resume_thunk: i64,
    suspend_thunk: i64,
    /// The `gc.roots` thunk (conservative root enumeration over the live fiber stacks). `0` when the
    /// module uses no fibers / `gc.roots`, or the target has no stack-switch support.
    roots_thunk: i64,
}

impl FiberEnv {
    fn null() -> FiberEnv {
        FiberEnv {
            new_thunk: 0,
            resume_thunk: 0,
            suspend_thunk: 0,
            roots_thunk: 0,
        }
    }
}

/// The §12 thread scheduler address + the two thunk addresses, baked into `thread.spawn`/`thread.join`
/// sites as constants. All `0` when the module uses no threads or the target has no stack-switch
/// support (in which case `ensure_supported` has already rejected any thread op).
#[derive(Clone, Copy)]
struct ThreadEnv {
    sched_addr: i64,
    spawn_thunk: i64,
    join_thunk: i64,
    wait_thunk: i64,
    notify_thunk: i64,
}

impl ThreadEnv {
    fn null() -> ThreadEnv {
        ThreadEnv {
            sched_addr: 0,
            spawn_thunk: 0,
            join_thunk: 0,
            wait_thunk: 0,
            notify_thunk: 0,
        }
    }
}

/// The §14 nesting runtime address + the `instantiate`/`join` thunk addresses, baked into the
/// module's `Instantiator` `cap.call` sites. All `0` when the module holds no `Instantiator`, or in a
/// **child** compilation (a JIT child cannot itself nest yet — its `Instantiator` cap.call falls
/// through to the ordinary `cap.call` path, i.e. an inert `CapFault`).
#[derive(Clone, Copy)]
struct InstEnv {
    nursery_addr: i64,
    instantiate_thunk: i64,
    join_thunk: i64,
    coro_spawn_thunk: i64,
    coro_resume_thunk: i64,
    // PROCESS.md S3 lifecycle thunks (poll / detach / kill) — parity with the interpreter's ops 9/10/12.
    poll_thunk: i64,
    detach_thunk: i64,
    kill_thunk: i64,
    // PROCESS.md S2 grant thunks — parity with the interpreter's re-grant of caps into a child's
    // powerbox: op 8 (`instantiate_granted`, single positional cap) and op 11 (`instantiate_named`,
    // multi-cap by name).
    instantiate_granted_thunk: i64,
    instantiate_named_thunk: i64,
    // STAGE1.md — op 13 (`instantiate_module_named`): run a separate `Module` *and* re-grant caps by
    // name (the shell "exec" primitive — union of op 5 + op 11).
    instantiate_module_named_thunk: i64,
}

impl InstEnv {
    fn null() -> InstEnv {
        InstEnv {
            nursery_addr: 0,
            instantiate_thunk: 0,
            join_thunk: 0,
            coro_spawn_thunk: 0,
            coro_resume_thunk: 0,
            poll_thunk: 0,
            detach_thunk: 0,
            kill_thunk: 0,
            instantiate_granted_thunk: 0,
            instantiate_named_thunk: 0,
            instantiate_module_named_thunk: 0,
        }
    }
    /// True when this compilation may lower `Instantiator` cap.calls to the nesting runtime (the
    /// parent compile with a live `Nursery`); `false` ⇒ they take the ordinary `cap.call` path.
    fn is_active(&self) -> bool {
        self.nursery_addr != 0
    }
}

/// The per-run `setjmp` table address + the four thunk/libc addresses baked into the module's
/// `SetJmp`/`LongJmp` sites (LLVM.md §"JIT `longjmp`", Option B). All `0` (`null`) when the module uses
/// no `setjmp`, or the target lacks the `setjmp_rt` runtime (non-unix / non-`fiber_rt`), in which case
/// `ensure_supported` has already rejected any `SetJmp`/`LongJmp`. `rt_addr` is the per-run
/// `SetjmpRuntime` (owned by the `CompiledModule`); `setjmp_addr`/`longjmp_addr` are libc `_setjmp`/
/// `_longjmp`, called inline in the guest frame.
#[derive(Clone, Copy)]
struct SetjmpEnv {
    rt_addr: i64,
    slot_thunk: i64,
    lookup_thunk: i64,
    setjmp_addr: i64,
    longjmp_addr: i64,
}

impl SetjmpEnv {
    fn null() -> SetjmpEnv {
        SetjmpEnv {
            rt_addr: 0,
            slot_thunk: 0,
            lookup_thunk: 0,
            setjmp_addr: 0,
            longjmp_addr: 0,
        }
    }
}

/// Per-function lowering context shared across blocks.
struct Lower<'a> {
    /// Holds `mem_base` (the window base) for load/store lowering and call threading.
    mem_var: Variable,
    /// Holds `fn_table_base` for `call_indirect` dispatch and call threading.
    fn_table_var: Variable,
    /// Holds `trap_out`, the host-owned `*mut i64` trap cell a trap (or the cap thunk)
    /// writes before returning (the host reads it to learn the run trapped, §5).
    trap_var: Variable,
    /// §2b path B: holds `stack_limit` (the running stack's `usable_low`, or 0 for the root), for the
    /// prologue [`emit_stack_check`] and to thread on to callees. Always present (the guard is in the
    /// always-on escape-TCB path).
    limit_var: Variable,
    /// This function's result CLIF types, so a trapping path can `return` dummy zeros.
    result_tys: Vec<Type>,
    /// The §4 confinement mask (`reserved - 1`); `0` when the module has no memory.
    mask: u64,
    /// The backed `mapped` extent in bytes — the guest window length handed to the `cap.call`
    /// thunk (`[mem_base, mem_base+mapped)`), so buffer borrows and Memory-cap ops bound against
    /// the *backed* region, not the larger reserved mask domain. `0` when the module has no memory.
    mapped: u64,
    /// The §14 nested sub-window base (`svm_mask::Window::sub`'s `base`): the masking lowering
    /// adds it to every confined address so the child lands in `[mem_base+base, …+reserved)`.
    /// `0` for an ordinary top-level window — the add is elided.
    sub_base: u64,
    /// Offset from `mem_base` of the **enclosing window's trailing guard page** —
    /// `round_up(win_reserved, page)`, where `win_reserved` is this window's own reservation for a
    /// top-level guest or the *parent's* reservation for a §14 sub-window. Always `PROT_NONE`, so the
    /// D63 branchless lowering redirects an out-of-bounds access here to fault it (for a sub-window the
    /// parent guard is the only guaranteed fault site — the child slice has parent memory on both
    /// sides). `0` when the module has no memory (the branchless path isn't taken).
    guard_offset: u64,
    /// The target's frontend config (pointer width + default call conv), for the Cranelift
    /// `call_memcpy`/`call_memmove`/`call_memset` helpers that lower the bulk-memory ops (D62).
    frontend_config: cranelift_codegen::isa::TargetFrontendConfig,
    /// The function-table index mask (`next_pow2(nfuncs) - 1`) for `call_indirect`.
    fn_table_mask: u64,
    /// The host `cap.call` thunk + ctx (constant addresses).
    cap: CapEnv,
    /// The §12 fiber runtime + thunk addresses for `cont.*` lowering.
    fiber: FiberEnv,
    /// The §12 thread scheduler + thunk addresses for `thread.*` lowering.
    thread: ThreadEnv,
    /// The §14 nesting runtime + thunk addresses for `Instantiator` `cap.call` lowering (`null` ⇒
    /// `Instantiator` cap.calls take the ordinary `cap.call` path — an inert `CapFault`).
    inst: InstEnv,
    /// The per-run `setjmp` table + libc `_setjmp`/`_longjmp` addresses for `SetJmp`/`LongJmp` lowering
    /// (`null` ⇒ the module has no `setjmp`, or the target lacks the runtime and `ensure_supported`
    /// already rejected the ops).
    setjmp: SetjmpEnv,
    /// Address of the host-owned **interrupt cell** (`AtomicU64`) for the §5 fuel/epoch kill-path.
    /// `0` ⇒ no kill-path is armed for this compile (the checks are not emitted — guest code is
    /// byte-identical to the un-armed build). When non-zero, the lowering polls `*epoch_addr` at
    /// loop back-edges and function entries and traps [`TrapKind::OutOfFuel`] if the host has set it
    /// non-zero, so a non-terminating guest is stopped. The guest cannot disable the poll — only the
    /// host (who chose to arm it) writes the cell.
    epoch_addr: i64,
    /// Every function's `FuncId`, so `call`/`return_call` can reference callees.
    ids: &'a [FuncId],
    /// The functions of this compilation unit, indexed like [`Self::ids`], so a **direct** `call`
    /// can read its callee's result types to decide the sret ABI (see [`uses_sret`]). `call_indirect`
    /// uses its own carried type instead.
    funcs: &'a [Func],
    /// Distinct module signatures, for `call_indirect` type ids.
    distinct: &'a [FuncType],
    /// The current function's **return-area pointer** variable when it returns via sret
    /// ([`uses_sret`] of its results), else `None`. A `Return` stores results through it; a tail
    /// call forwards it (the tail callee shares the caller's result type, so its sret-ness matches).
    sret_var: Option<Variable>,
    /// `-g` source-loc lookup (W5 JIT/DWARF Stage 0): `(func, block, inst) → debug_info.locs` index.
    /// `None` ⇒ no debug info, so no `SourceLoc`s are stamped (codegen is byte-identical to before).
    srclocs: Option<&'a SrcLocMap>,
    /// This function's index, the `func` half of the [`Self::srclocs`] key.
    func_idx: u32,
    /// `-g` value-label points (W5 JIT/DWARF Stage 3a): `(func, block) → [(block-local value, label)]`
    /// — the source variables whose backing CLIF value `lower_block` stamps with a `ValueLabel`, so
    /// Cranelift records its machine-location ranges. `None` ⇒ no `-g` vars (no labels, codegen
    /// unchanged).
    var_labels: Option<&'a VarLabelMap>,
    /// §5 W3 Stage 2 explicit-trap backtrace: `Some((sigref, addr))` to bake a call to the
    /// trap-capture helper (`svm_capture_explicit_trap`) into every [`emit_trap`] site, so an
    /// explicit trap (div-by-zero, `unreachable`, `OutOfFuel`, indirect-call-type) records a source
    /// backtrace before it unwinds — the way memory faults do via the signal handler. `None` without
    /// `-g` (the backtrace would be empty) or where the helper isn't linked (a target with no trap
    /// runtime), leaving the trap path byte-identical to before.
    trap_capture: Option<(SigRef, i64)>,
}

/// The address of the §5 W3 explicit-trap capture helper (`svm_capture_explicit_trap` in
/// `trap_capture.c`): baked into each [`emit_trap`] site under `-g`. The helper takes the trapping
/// frame pointer as an argument (the JIT threads it in via Cranelift `get_frame_pointer`), so it works
/// on every target the shim compiles for — **unix and windows** (MSVC has no `__builtin_frame_address`,
/// but the passed `fp` + `_ReturnAddress` sidestep it). `0` on a target with no trap runtime.
#[cfg(any(unix, windows))]
fn trap_capture_addr() -> i64 {
    extern "C" {
        fn svm_capture_explicit_trap(fp: usize);
    }
    svm_capture_explicit_trap as unsafe extern "C" fn(usize) as usize as i64
}
#[cfg(not(any(unix, windows)))]
fn trap_capture_addr() -> i64 {
    0
}

/// Build the natural-ABI CLIF for one IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. The CLIF entry block holds the native params and jumps into IR
/// block 0 passing the parameters as its block args.
///
/// `fn_table_mask` is the `call_indirect` index mask — `next_pow2(table_len) - 1` for the
/// table this function will dispatch through. It is an explicit parameter (not derived from
/// `ids.len()`) because an **incrementally defined** function (`CompiledModule::define_extra`)
/// has its own unit-local `ids` for direct calls but dispatches through the *parent's*
/// function table, whose mask was fixed when the parent was compiled (the mask is baked as a
/// constant into every call site — DESIGN.md §22 "the baked function-table mask").
#[allow(clippy::too_many_arguments)]
fn build_clif(
    module: &mut JITModule,
    ids: &[FuncId],
    funcs: &[Func],
    distinct: &[FuncType],
    cap: CapEnv,
    fiber: FiberEnv,
    thread: ThreadEnv,
    inst: InstEnv,
    setjmp: SetjmpEnv,
    clif: &mut Function,
    f: &Func,
    mask: u64,
    mapped: u64,
    sub_base: u64,
    guard_offset: u64,
    epoch_addr: i64,
    fn_table_mask: u64,
    func_idx: u32,
    srclocs: Option<&SrcLocMap>,
    var_labels: Option<&VarLabelMap>,
) -> Result<(), JitError> {
    if f.blocks.is_empty() {
        return Err(JitError::Malformed);
    }
    clif.signature = natural_sig(module, f);
    clif.name = UserFuncName::user(0, 0);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);

    // One CLIF block per IR block, with params mirroring the IR block params. A
    // separate CLIF entry block holds the native params and jumps into IR block 0.
    let blocks: Vec<_> = f.blocks.iter().map(|_| b.create_block()).collect();
    for (i, blk) in f.blocks.iter().enumerate() {
        for p in &blk.params {
            b.append_block_param(blocks[i], clif_ty(*p));
        }
    }
    let sret = uses_sret(&f.results);
    let entry = b.create_block();
    b.append_block_param(entry, I64); // mem_base
    b.append_block_param(entry, I64); // fn_table_base
    b.append_block_param(entry, I64); // trap_out
    b.append_block_param(entry, I64); // stack_limit (§2b path B) — mirrors sig_from
    if sret {
        b.append_block_param(entry, I64); // return-area pointer (results spilled here, not returned)
    }
    for p in &f.params {
        b.append_block_param(entry, clif_ty(*p));
    }
    b.switch_to_block(entry);
    b.seal_block(entry);
    let mem_base = b.block_params(entry)[0];
    let fn_table_base = b.block_params(entry)[1];
    let trap_out = b.block_params(entry)[2];

    // The context pointers are needed across blocks; stash them in variables.
    let mem_var = b.declare_var(I64);
    b.def_var(mem_var, mem_base);
    let fn_table_var = b.declare_var(I64);
    b.def_var(fn_table_var, fn_table_base);
    let trap_var = b.declare_var(I64);
    b.def_var(trap_var, trap_out);
    // §2b path B: stash the stack-limit param (block index 3, right after the context pointers) so the
    // prologue check and every call can reach it.
    let limit_var = {
        let var = b.declare_var(I64);
        b.def_var(var, b.block_params(entry)[3]);
        var
    };
    // The return-area pointer (when sret) is likewise needed in the `Return`/tail-call blocks. Its
    // block index is after the context pointers + the stack-limit param (index 4).
    let sret_var = if sret {
        let var = b.declare_var(I64);
        b.def_var(var, b.block_params(entry)[4]);
        Some(var)
    } else {
        None
    };
    // §5 W3 Stage 2: under `-g`, import the trap-capture helper's `(i64) -> ()` host-C signature (it
    // takes the trapping frame pointer) and bake its address, so `emit_trap` can record an explicit
    // trap's source backtrace before unwinding. Disabled without `-g` or where the helper isn't linked.
    let trap_capture = match trap_capture_addr() {
        addr if srclocs.is_some() && addr != 0 => {
            let mut sig = module.make_signature(); // host C ABI
            sig.params.push(AbiParam::new(I64)); // the trapping frame pointer
            Some((b.import_signature(sig), addr))
        }
        _ => None,
    };
    let lower = Lower {
        mem_var,
        fn_table_var,
        trap_var,
        limit_var,
        sret_var,
        funcs,
        result_tys: f.results.iter().map(|t| clif_ty(*t)).collect(),
        mask,
        mapped,
        sub_base,
        guard_offset,
        frontend_config: module.target_config(),
        fn_table_mask,
        cap,
        fiber,
        thread,
        inst,
        setjmp,
        epoch_addr,
        ids,
        distinct,
        srclocs,
        func_idx,
        var_labels,
        trap_capture,
    };

    // Jump into IR block 0 passing the function parameters (entry params after the three context
    // pointers, plus the sret pointer when present). A §5 kill-path check guards the *entry* (caught
    // before any work): this is what stops unbounded recursion and tail-call loops — each (re-)entry
    // polls the interrupt cell. Intra-function loops are caught by the per-back-edge check in
    // `lower_block`.
    let pbase = 4 + usize::from(sret); // 3 context ptrs + stack_limit, then sret
    let entry_args: Vec<BlockArg> = b.block_params(entry)[pbase..]
        .iter()
        .map(|v| BlockArg::from(*v))
        .collect();
    emit_epoch_check(&mut b, &lower);
    emit_stack_check(&mut b, &lower);
    b.ins().jump(blocks[0], &entry_args);

    for (i, blk) in f.blocks.iter().enumerate() {
        lower_block(module, &mut b, blk, i, blocks[i], &blocks, &lower)?;
    }

    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

/// Build the fixed buffer-ABI trampoline `fn(args_ptr, results_ptr, mem_base,
/// fn_table_base, trap_out)` that decodes the entry function's args from `args_ptr`,
/// calls it (natural ABI), and stores its results to `results_ptr`. This is what Rust
/// calls, so any arity works.
fn build_trampoline(module: &mut JITModule, clif: &mut Function, entry_id: FuncId, entry: &Func) {
    clif.signature.params.push(AbiParam::new(I64)); // args_ptr
    clif.signature.params.push(AbiParam::new(I64)); // results_ptr
    clif.signature.params.push(AbiParam::new(I64)); // mem_base
    clif.signature.params.push(AbiParam::new(I64)); // fn_table_base
    clif.signature.params.push(AbiParam::new(I64)); // trap_out
    clif.name = UserFuncName::user(0, 1);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    for _ in 0..5 {
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(blk);
    b.seal_block(blk);
    let args_ptr = b.block_params(blk)[0];
    let results_ptr = b.block_params(blk)[1];
    let mem_base = b.block_params(blk)[2];
    let fn_table_base = b.block_params(blk)[3];
    let trap_out = b.block_params(blk)[4];

    // Decode args (context pointers first), call the entry, store results. When the entry returns
    // via sret, hand it `results_ptr` directly as the return-area pointer — it writes its results
    // (8-byte `encode_slot` slots) straight into the buffer Rust reads, so no register read-back.
    let sret = uses_sret(&entry.results);
    let mut call_args = vec![mem_base, fn_table_base, trap_out];
    // §2b path B: the root runs on the OS thread stack (OS-guarded), so its stack-limit is 0 ⇒ the
    // prologue check is inert for the root computation; fibers get a real limit at their own entry.
    let zero = b.ins().iconst(I64, 0);
    call_args.push(zero); // stack_limit = 0 (root)
    if sret {
        call_args.push(results_ptr);
    }
    for (i, p) in entry.params.iter().enumerate() {
        let slot = b
            .ins()
            .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        call_args.push(decode_slot(&mut b, slot, *p));
    }
    let callee = module.declare_func_in_func(entry_id, b.func);
    let call = b.ins().call(callee, &call_args);
    if !sret {
        let rets: Vec<Value> = b.inst_results(call).to_vec();
        for (i, r) in rets.iter().enumerate() {
            let slot = encode_slot(&mut b, *r);
            b.ins()
                .store(MemFlags::trusted(), slot, results_ptr, (i * 8) as i32);
        }
    }
    b.ins().return_(&[]);
    b.seal_all_blocks();
    b.finalize();
}

/// The fixed signature of a §12 fiber/thread entry: `(i64 sp, i64 arg) -> i64` (the unified
/// frontend convention). Its structural id is what a `cont.new` funcref is type-checked against.
#[cfg(fiber_rt)]
fn fiber_func_type() -> FuncType {
    FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    }
}

/// Whether `m` contains any fiber op, so `run_inner` knows to stand up the fiber runtime.
#[cfg(fiber_rt)]
fn module_uses_fibers(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| {
        f.blocks.iter().any(|blk| {
            blk.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ContNew { .. }
                        | Inst::ContResume { .. }
                        | Inst::Suspend { .. }
                        // `gc.roots` walks the fiber runtime's live stacks, so it needs the runtime
                        // stood up even if the module never explicitly creates a fiber.
                        | Inst::GcRoots { .. }
                )
            })
        })
    })
}

/// Whether `m` contains any thread op (spawn/join/wait/notify), so `run_inner` knows to run under the
/// thread scheduler.
#[cfg(fiber_rt)]
fn module_uses_threads(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| {
        f.blocks.iter().any(|blk| {
            blk.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ThreadSpawn { .. }
                        | Inst::ThreadJoin { .. }
                        | Inst::MemoryWait { .. }
                        | Inst::MemoryNotify { .. }
                )
            })
        })
    })
}

/// Whether `m` contains any `setjmp`/`longjmp` op, so `run_inner` knows to stand up the per-run
/// [`setjmp_rt::SetjmpRuntime`] whose address is baked into those sites.
#[cfg(setjmp_rt)]
fn module_uses_setjmp(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| {
        f.blocks.iter().any(|blk| {
            blk.insts
                .iter()
                .any(|i| matches!(i, Inst::SetJmp { .. } | Inst::LongJmp { .. }))
        })
    })
}

/// Whether `m` holds a §14 `Instantiator` — a `cap.call` to iface 6 (`svm_interp::iface::INSTANTIATOR`)
/// — so `run_inner` knows to stand up the nesting [`instantiator_rt::Nursery`].
#[cfg(fiber_rt)]
fn module_uses_instantiator(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| {
        f.blocks.iter().any(|blk| {
            blk.insts
                .iter()
                .any(|i| matches!(i, Inst::CapCall { type_id: 6, .. }))
        })
    })
}

/// Build the generic fiber **call-trampoline**: `extern "C" fn(code, mem_base, fn_table_base,
/// trap_out, sp, arg) -> i64` that `call_indirect`s a guest fiber entry under its `Tail` ABI. Rust
/// cannot call a `Tail`-convention function directly, so the fiber runtime calls this (default C ABI)
/// instead; one trampoline serves all fibers since every entry is `(i64 sp, i64 arg) -> i64`.
#[cfg(fiber_rt)]
fn build_fiber_call_trampoline(module: &mut JITModule, clif: &mut Function) {
    // Params: `code`, then the guest entry's Tail args in order — mem_base, fn_table_base, trap_out,
    // stack_limit (§2b path B, always present), sp, arg -> i64. `stack_limit` sits in the same relative
    // slot as in `sig_from`, so the entry call args are exactly `p[1..]` (see [`FiberCallTramp`]).
    let np = 7; // code + (mem_base, fn_table_base, trap_out, stack_limit, sp, arg)
    for _ in 0..np {
        clif.signature.params.push(AbiParam::new(I64));
    }
    clif.signature.returns.push(AbiParam::new(I64));
    clif.name = UserFuncName::user(0, 2);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    for _ in 0..np {
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(blk);
    b.seal_block(blk);
    let p = b.block_params(blk).to_vec();
    let code = p[0];
    let call_args = p[1..].to_vec(); // (mem_base, fn_table_base, trap_out, [stack_limit], sp, arg)
    let sig = b.import_signature(sig_from(
        module,
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
    ));
    let call = b.ins().call_indirect(sig, code, &call_args);
    let r = b.inst_results(call)[0];
    b.ins().return_(&[r]);
    b.seal_all_blocks();
    b.finalize();
}

/// Lower one IR block's body + terminator into its CLIF block.
fn lower_block(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    blk: &Block,
    block_idx: usize,
    cb: cranelift_codegen::ir::Block,
    blocks: &[cranelift_codegen::ir::Block],
    lower: &Lower,
) -> Result<(), JitError> {
    b.switch_to_block(cb);
    // The CLIF block params are the IR block params; seed the value map with them.
    let mut vals: Vec<Value> = b.block_params(cb).to_vec();
    // Parallel upper-bound map (for mask elision); block params are unknown. Kept in lockstep
    // with `vals` so `ubs[i]` always describes IR value `i` (a misalignment could mis-elide,
    // so it is grown at the same points `vals` is). `size` confines via `mask` (= size−1).
    let mut ubs: Vec<u64> = vec![UB_TOP; vals.len()];
    let size = lower.mask.wrapping_add(1);

    for (inst_idx, inst) in blk.insts.iter().enumerate() {
        // W5 JIT/DWARF Stage 0: stamp the ops this IR instruction lowers to with a `SourceLoc`
        // (= its `debug_info.locs` index), so the finalized code's address map carries source
        // positions. Only when the module has `-g` debug info; otherwise codegen is unchanged.
        if let Some(map) = lower.srclocs {
            if let Some(&loc) = map.get(&(lower.func_idx, block_idx as u32, inst_idx as u32)) {
                b.set_srcloc(SourceLoc::new(loc));
            }
        }
        // `call`/`call_indirect` append 0..N results — handle before the single-value
        // match (which produces exactly one value).
        if let Inst::Call { func, args } = inst {
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let results = &lower
                .funcs
                .get(*func as usize)
                .ok_or(JitError::Malformed)?
                .results;
            let mut cargs = ctx_args(b, lower);
            let sret = sret_call_slot(b, &mut cargs, results);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call(callee, &cargs);
            // A trap raised inside the callee leaves the trap cell set and returns zeros; propagate
            // it here so it unwinds immediately (else the caller would run on with bogus results,
            // and a later successful `cap.call` could reset the cell, masking the trap). On the sret
            // path this also returns *before* reading the unwritten return-area slots.
            emit_trap_propagate(b, lower);
            read_call_results(b, call, sret, results, &mut vals);
            ubs.resize(vals.len(), UB_TOP); // call results are unknown
            continue;
        }
        if let Inst::CallIndirect { ty, idx, args } = inst {
            let code = indirect_dispatch(b, lower, get(&vals, *idx)?, ty);
            let sig = b.import_signature(sig_from(module, &ty.params, &ty.results));
            let mut cargs = ctx_args(b, lower);
            let sret = sret_call_slot(b, &mut cargs, &ty.results);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call_indirect(sig, code, &cargs);
            // Propagate a callee trap immediately (see the direct-call case above).
            emit_trap_propagate(b, lower);
            read_call_results(b, call, sret, &ty.results, &mut vals);
            ubs.resize(vals.len(), UB_TOP); // call results are unknown
            continue;
        }
        if let Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } = inst
        {
            // §14 `Instantiator` (iface 6): when this (parent) compile has a live `Nursery`, lower
            // `instantiate`/`join` to its thunks instead of the generic `cap.call` — spawning a child
            // needs the host compiler, which the flat `cap.call` thunk can't reach. Otherwise (a child
            // compile, or no nesting runtime) it falls through to the ordinary path (an inert CapFault).
            if *type_id == 6 && lower.inst.is_active() {
                lower_instantiator(module, b, lower, *op, sig, *handle, args, &mut vals)?;
            } else if let Some(target) = fast_cap_target(lower, *type_id, *op, sig) {
                // D45 devirtualized fast path: a register-to-register direct call to the specialized
                // host fn the resolver claimed for this `(type_id, op)`.
                lower_cap_call_fast(module, b, lower, target, sig, *handle, args, &mut vals)?;
            } else {
                let h = get(&vals, *handle)?;
                lower_cap_call(module, b, lower, *type_id, *op, sig, h, args, &mut vals)?;
            }
            ubs.resize(vals.len(), UB_TOP); // cap-call results are unknown
            continue;
        }
        // §7 executable named import (IMPORTS.md phase 1): lower like the `cap.self.*` family — a
        // `cap.call` thunk with the reserved `CAP_IMPORT_TYPE_ID` and the **import index** as the
        // op. The host's dispatch translates it through the instantiation-time binding table
        // (import `i` → bound `(type_id, op)` + granted handle) and re-dispatches — one shared
        // implementation with the interpreter and the bytecode engine. The vestigial handle operand
        // is not read (constant 0, like `cap.self.*`); the module bytes are never rewritten, so the
        // compiled code is identical across instantiations (the binding is host-side state).
        if let Inst::CallImport {
            import, sig, args, ..
        } = inst
        {
            let h0 = b.ins().iconst(I32, 0);
            lower_cap_call(
                module,
                b,
                lower,
                svm_ir::CAP_IMPORT_TYPE_ID,
                *import,
                sig,
                h0,
                args,
                &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // Phase-2 `import.attach` (IMPORTS.md): the attach sentinel with the handle value as the
        // one `i32` argument; `i32` status result. Same shared host entry as the interpreter and
        // the bytecode engine.
        if let Inst::ImportAttach { import, handle } = inst {
            let h0 = b.ins().iconst(I32, 0);
            let sig = FuncType {
                params: vec![ValType::I32],
                results: vec![ValType::I32],
            };
            let call_args = [*handle];
            lower_cap_call(
                module,
                b,
                lower,
                svm_ir::CAP_IMPORT_ATTACH_TYPE_ID,
                *import,
                &sig,
                h0,
                &call_args,
                &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §7/§6 capability reflection: lower to a `cap.call` thunk with the reserved `CAP_SELF_TYPE_ID`
        // (op 0 = count, op 1 = get, op 4 = attest) — the host services it directly, matching the
        // interpreter. The handle is unused there, so pass a constant 0.
        if matches!(
            inst,
            Inst::CapSelfCount | Inst::CapSelfGet { .. } | Inst::CapSelfAttest
        ) {
            let h0 = b.ins().iconst(I32, 0);
            let (op, sig, call_args): (u32, FuncType, &[u32]) = match inst {
                Inst::CapSelfCount => (
                    0,
                    FuncType {
                        params: vec![],
                        results: vec![ValType::I32],
                    },
                    &[],
                ),
                // §6 `cap.self.attest` — op 4, no args, one packed `i32` (the non-interposable trust
                // anchor; the child host's provenance the thunk reports).
                Inst::CapSelfAttest => (
                    4,
                    FuncType {
                        params: vec![],
                        results: vec![ValType::I32],
                    },
                    &[],
                ),
                Inst::CapSelfGet { idx } => (
                    1,
                    FuncType {
                        params: vec![ValType::I32],
                        results: vec![ValType::I32, ValType::I32],
                    },
                    std::slice::from_ref(idx),
                ),
                _ => unreachable!(),
            };
            lower_cap_call(
                module,
                b,
                lower,
                svm_ir::CAP_SELF_TYPE_ID,
                op,
                &sig,
                h0,
                call_args,
                &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §7 `cap.self.resolve` — op 2 over the reserved `CAP_SELF_TYPE_ID`, with a `(name_ptr,
        // name_len)` buffer (the thunk reads the window to resolve the name, like any cap op). One
        // i32 result; the handle operand is unused (constant 0), as for count/get.
        if let Inst::CapSelfResolve { name_ptr, name_len } = inst {
            let h0 = b.ins().iconst(I32, 0);
            let sig = FuncType {
                params: vec![ValType::I64, ValType::I64],
                results: vec![ValType::I32],
            };
            let call_args = [*name_ptr, *name_len];
            lower_cap_call(
                module,
                b,
                lower,
                svm_ir::CAP_SELF_TYPE_ID,
                2,
                &sig,
                h0,
                &call_args,
                &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §7 `cap.self.label` — op 3 over `CAP_SELF_TYPE_ID`: `(handle, buf_ptr, buf_cap)` → label len
        // (the thunk writes the label into the window). One i32 result; cap.call handle unused (0).
        if let Inst::CapSelfLabel {
            handle,
            buf_ptr,
            buf_cap,
        } = inst
        {
            let h0 = b.ins().iconst(I32, 0);
            let sig = FuncType {
                params: vec![ValType::I32, ValType::I64, ValType::I64],
                results: vec![ValType::I32],
            };
            let call_args = [*handle, *buf_ptr, *buf_cap];
            lower_cap_call(
                module,
                b,
                lower,
                svm_ir::CAP_SELF_TYPE_ID,
                3,
                &sig,
                h0,
                &call_args,
                &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §12 fibers: lower `cont.*` to indirect calls to the host fiber thunks (addresses baked into
        // `lower.fiber`), threading `mem_base`/`fn_table_base`/`trap_out` like `cap.call`. A thunk that
        // sets the trap cell (forged handle, bad funcref, fiber-bomb, root suspend) propagates here.
        if let Inst::ContNew { func, sp } = inst {
            // fiber_new(mem_base, fn_table_base, trap_out, funcref:i32, sp:i64) -> i32 handle. The
            // running vCPU's fiber runtime is read from a thread-local, so threads + fibers compose.
            let mem_base = b.use_var(lower.mem_var);
            let fnt = b.use_var(lower.fn_table_var);
            let trap_out = b.use_var(lower.trap_var);
            let funcref = get(&vals, *func)?;
            let spv = get(&vals, *sp)?;
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64)); // i64 fiber handle (16-bit slot + 48-bit generation)
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.fiber.new_thunk);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[mem_base, fnt, trap_out, funcref, spv]);
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::ContResume { k, arg } = inst {
            // fiber_resume(handle:i64, arg:i64, status_out:*i64, trap_out:i64) -> value:i64.
            // Results are appended (status:i32, value:i64) to match the IR's two-result shape.
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let kh = get(&vals, *k)?;
            let av = get(&vals, *arg)?;
            let trap_out = b.use_var(lower.trap_var);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.fiber.resume_thunk);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[kh, av, status_ptr, trap_out]);
            emit_trap_propagate(b, lower);
            let value = b.inst_results(call)[0];
            let status64 = b.ins().stack_load(I64, ss, 0);
            let status = b.ins().ireduce(I32, status64);
            vals.push(status); // result 0: status (i32)
            vals.push(value); // result 1: value (i64)
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::Suspend { value } = inst {
            // fiber_suspend(value:i64, trap_out:i64) -> next-resume arg:i64
            let v = get(&vals, *value)?;
            let trap_out = b.use_var(lower.trap_var);
            let mut tsig = module.make_signature();
            for t in [I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.fiber.suspend_thunk);
            let call = b.ins().call_indirect(tref, thunk, &[v, trap_out]);
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // `setjmp` (LLVM.md §"JIT `longjmp`", Option B): two calls. First a host thunk returns the
        // stable host `jmp_buf` slot for this guest `buf` address (`rt_setjmp_slot`); then `_setjmp` is
        // called **inline in this guest frame** — the frame a later `longjmp` returns to — so it saves
        // *this* frame's SP/return-addr. The libc `_setjmp` address is baked directly (not wrapped in a
        // Rust thunk, whose frame would be gone by `longjmp` time — UB). Result is the `i32` 0 (direct)
        // / long-jump value (re-entry). The slot alloc is infallible, so no trap-propagate.
        if let Inst::SetJmp { buf } = inst {
            let bufv = get(&vals, *buf)?; // i64 guest jmp_buf window address (the table key)
                                          // slot = rt_setjmp_slot(rt_addr, buf) -> *mut jmp_buf
            let mut s1 = module.make_signature();
            for t in [I64, I64] {
                s1.params.push(AbiParam::new(t));
            }
            s1.returns.push(AbiParam::new(I64));
            let r1 = b.import_signature(s1);
            let slot_thunk = b.ins().iconst(I64, lower.setjmp.slot_thunk);
            let rt_addr = b.ins().iconst(I64, lower.setjmp.rt_addr);
            let c1 = b.ins().call_indirect(r1, slot_thunk, &[rt_addr, bufv]);
            let slot = b.inst_results(c1)[0];
            // r = _setjmp(slot) -> i32 — emitted inline in this JIT frame.
            let mut s2 = module.make_signature();
            s2.params.push(AbiParam::new(I64));
            s2.returns.push(AbiParam::new(I32));
            let r2 = b.import_signature(s2);
            let setjmp_fn = b.ins().iconst(I64, lower.setjmp.setjmp_addr);
            let c2 = b.ins().call_indirect(r2, setjmp_fn, &[slot]);
            vals.push(b.inst_results(c2)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // `longjmp` (LLVM.md §"JIT `longjmp`"): look up the host `jmp_buf` slot for this `buf` (set by a
        // prior `setjmp`); a miss writes the trap cell and `emit_trap_propagate` bails *before* the
        // (skipped) `_longjmp`. Otherwise `_longjmp(slot, val)` restores the saved frame and never
        // returns — the IR's trailing `unreachable` terminator caps the block (the call isn't marked
        // noreturn, but the dead fall-through is terminated by it). No result.
        if let Inst::LongJmp { buf, val } = inst {
            let bufv = get(&vals, *buf)?;
            let valv = get(&vals, *val)?; // i32 long-jump value (0 → 1 is applied by libc `_longjmp`)
            let trap_out = b.use_var(lower.trap_var);
            // slot = rt_setjmp_lookup(rt_addr, buf, trap_out) -> *mut jmp_buf (null + trap on miss)
            let mut s1 = module.make_signature();
            for t in [I64, I64, I64] {
                s1.params.push(AbiParam::new(t));
            }
            s1.returns.push(AbiParam::new(I64));
            let r1 = b.import_signature(s1);
            let lookup_thunk = b.ins().iconst(I64, lower.setjmp.lookup_thunk);
            let rt_addr = b.ins().iconst(I64, lower.setjmp.rt_addr);
            let c1 = b
                .ins()
                .call_indirect(r1, lookup_thunk, &[rt_addr, bufv, trap_out]);
            let slot = b.inst_results(c1)[0];
            emit_trap_propagate(b, lower); // bail on a miss (forged/stale token) before `_longjmp`
                                           // _longjmp(slot, val) — inline, noreturn.
            let mut s2 = module.make_signature();
            s2.params.push(AbiParam::new(I64));
            s2.params.push(AbiParam::new(I32));
            let r2 = b.import_signature(s2);
            let longjmp_fn = b.ins().iconst(I64, lower.setjmp.longjmp_addr);
            b.ins().call_indirect(r2, longjmp_fn, &[slot, valv]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §12 per-vCPU TLS register: baked thunks over a per-OS-thread word (no window/trap context —
        // a pure thread-local access that cannot fault). `get` reads the *current* vCPU's word, so it
        // is correct after a fiber migrates here.
        if let Inst::VcpuTlsGet = inst {
            let mut tsig = module.make_signature();
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, vcpu_tls::get as *const () as i64);
            let call = b.ins().call_indirect(tref, thunk, &[]);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §12.8 4A.5 durable-runtime-internal: read the current context's shadow-region base from the
        // per-OS-thread register (a baked thunk, like `vcpu.tls.get`; cannot fault, no window/trap
        // context). The durable transform emits this to address this context's own shadow-SP word.
        if let Inst::DurableShadowBase = inst {
            let mut tsig = module.make_signature();
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, durable_shadow::get as *const () as i64);
            let call = b.ins().call_indirect(tref, thunk, &[]);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::VcpuTlsSet { val } = inst {
            let v = get(&vals, *val)?;
            let mut tsig = module.make_signature();
            tsig.params.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, vcpu_tls::set as *const () as i64);
            b.ins().call_indirect(tref, thunk, &[v]);
            continue;
        }
        if let Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } = inst
        {
            // gc_roots(heap_lo, heap_hi, payload_mask, buf, cap, mem_base, mask, mapped, sub_base,
            // trap_out) -> i64 count. The thunk walks the live fiber stacks (runtime via the
            // thread-local), masks each word with `payload_mask` (§GC tagged pointers; distinct from
            // the window-confinement `mask`), filters the masked value to `[heap_lo, heap_hi)`, and
            // writes the first `cap` deduped words to guest `buf` — confining/bounds-checking it with
            // the same `mask`/`mapped`/`sub_base` as `mask_addr`, so a forged buffer faults (below).
            let lo = get(&vals, *heap_lo)?;
            let hi = get(&vals, *heap_hi)?;
            let payload_mask = get(&vals, *mask)?;
            let dst = get(&vals, *buf)?;
            let cap_v = get(&vals, *cap)?;
            let mem_base = b.use_var(lower.mem_var);
            let maskv = b.ins().iconst(I64, lower.mask as i64);
            let mappedv = b.ins().iconst(I64, lower.mapped as i64);
            let subv = b.ins().iconst(I64, lower.sub_base as i64);
            let trap_out = b.use_var(lower.trap_var);
            // Marshal the ten args into one 8-byte-aligned stack slot (matching `GcRootsArgs`'s
            // `#[repr(C)]` field order) and pass a single pointer. The thunk is the register-flush
            // trampoline, which spills the callee-saved registers before the scan; a one-pointer ABI
            // lets it do so with no per-argument reshuffle (see `fiber_rt::svm_gc_roots_flush`).
            let slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                80, // 10 × 8 bytes
                3,  // 8-byte aligned
            ));
            for (i, v) in [
                lo,
                hi,
                payload_mask,
                dst,
                cap_v,
                mem_base,
                maskv,
                mappedv,
                subv,
                trap_out,
            ]
            .into_iter()
            .enumerate()
            {
                b.ins().stack_store(v, slot, (i * 8) as i32);
            }
            let args_ptr = b.ins().stack_addr(I64, slot, 0);
            let mut tsig = module.make_signature();
            tsig.params.push(AbiParam::new(I64)); // args pointer
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.fiber.roots_thunk);
            let call = b.ins().call_indirect(tref, thunk, &[args_ptr]);
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        // §12 threads: lower `thread.spawn`/`thread.join` to indirect calls to the host scheduler
        // thunks (addresses baked into `lower.thread`), threading `mem_base`/`fn_table_base`/`trap_out`
        // like `cap.call`. A thunk that sets the trap cell (forged handle, thread-bomb) propagates here.
        if let Inst::ThreadSpawn { func, sp, arg } = inst {
            // thread_spawn(sched, mem_base, fn_table_base, trap_out, func_idx:i32, sp:i64, arg:i64) -> i32
            let sched = b.ins().iconst(I64, lower.thread.sched_addr);
            let mem_base = b.use_var(lower.mem_var);
            let fnt = b.use_var(lower.fn_table_var);
            let trap_out = b.use_var(lower.trap_var);
            let func_idx = b.ins().iconst(I32, *func as i64);
            let spv = get(&vals, *sp)?;
            let av = get(&vals, *arg)?;
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I64, I32, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.thread.spawn_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[sched, mem_base, fnt, trap_out, func_idx, spv, av],
            );
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::ThreadJoin { handle } = inst {
            // thread_join(sched, handle:i32, trap_out:i64) -> result:i64
            let sched = b.ins().iconst(I64, lower.thread.sched_addr);
            let h = get(&vals, *handle)?;
            let trap_out = b.use_var(lower.trap_var);
            let mut tsig = module.make_signature();
            for t in [I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.thread.join_thunk);
            let call = b.ins().call_indirect(tref, thunk, &[sched, h, trap_out]);
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } = inst
        {
            // thread_wait(sched, phys:i64, expected:i64, width:i32, timeout:i64, trap_out:i64) ->
            // status:i32. `trap_out` carries the §12.8 thaw fail-closed (a re-issued wait that would
            // park on the single worker traps `ThreadFault`); on a fresh run it is never written.
            let w = atomic_width(*ty);
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false, w);
            guard_atomic_align(b, lower, phys, w); // misaligned wait traps (like the other atomics)
            let sched = b.ins().iconst(I64, lower.thread.sched_addr);
            let exp_raw = get(&vals, *expected)?;
            let exp = if w < 8 {
                b.ins().uextend(I64, exp_raw) // compare is bit-equality on the low `w` bytes
            } else {
                exp_raw
            };
            let width = b.ins().iconst(I32, w as i64);
            let to = get(&vals, *timeout)?;
            let trap_out = b.use_var(lower.trap_var);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I32, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.thread.wait_thunk);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[sched, phys, exp, width, to, trap_out]);
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::MemoryNotify { addr, count } = inst {
            // thread_notify(sched, phys:i64, count:i32) -> woken:i32. Accesses no memory (the address
            // is only confined, no alignment requirement — matching the interpreter).
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false, 4);
            let sched = b.ins().iconst(I64, lower.thread.sched_addr);
            let cnt = get(&vals, *count)?;
            let mut tsig = module.make_signature();
            for t in [I64, I64, I32] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.thread.notify_thunk);
            let call = b.ins().call_indirect(tref, thunk, &[sched, phys, cnt]);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        let v = match inst {
            Inst::ConstI32(c) => b.ins().iconst(I32, *c as i64),
            Inst::ConstI64(c) => b.ins().iconst(I64, *c),
            Inst::IntBin { ty, op, a, b: rb } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                match op {
                    // div/rem trap on a zero divisor (and signed div on INT_MIN/-1):
                    // guard with explicit checks that branch to a trap-return.
                    BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => {
                        lower_div_rem(b, lower, *ty, *op, x, y)
                    }
                    _ => int_bin(b, *op, x, y),
                }
            }
            Inst::IntUn { op, a, ty } => {
                let x = get(&vals, *a)?;
                let rt = int_clif_ty(*ty);
                // `extendN_s` sign-extends the low N bits: reduce to iN, then sign-extend
                // back to the (same) result width. When N == the result width (`extend32_s`
                // on `i32`) it's the identity — `ireduce`/`sextend` both require a strict
                // width change, so pass `x` through.
                let sext_low = |b: &mut FunctionBuilder, nt: Type| {
                    if nt == rt {
                        x
                    } else {
                        let r = b.ins().ireduce(nt, x);
                        b.ins().sextend(rt, r)
                    }
                };
                match op {
                    IntUnOp::Clz => b.ins().clz(x),
                    IntUnOp::Ctz => b.ins().ctz(x),
                    IntUnOp::Popcnt => b.ins().popcnt(x),
                    IntUnOp::Extend8S => sext_low(b, I8),
                    IntUnOp::Extend16S => sext_low(b, I16),
                    IntUnOp::Extend32S => sext_low(b, I32),
                }
            }
            Inst::IntCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().icmp(int_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Eqz { a, .. } => {
                let x = get(&vals, *a)?;
                let c = b.ins().icmp_imm(IntCC::Equal, x, 0);
                b.ins().uextend(I32, c)
            }
            Inst::Select { cond, a, b: rb } => {
                let (c, x, y) = (get(&vals, *cond)?, get(&vals, *a)?, get(&vals, *rb)?);
                b.ins().select(c, x, y)
            }
            Inst::ConstF32(bits) => {
                // Materialize via the exact bit pattern (NaN-safe), then bitcast.
                let i = b.ins().iconst(I32, *bits as i64);
                b.ins().bitcast(F32, MemFlags::new(), i)
            }
            Inst::ConstF64(bits) => {
                let i = b.ins().iconst(I64, *bits as i64);
                b.ins().bitcast(F64, MemFlags::new(), i)
            }
            Inst::FBin { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                float_bin(b, *op, x, y)
            }
            Inst::FUn { op, a, .. } => {
                let x = get(&vals, *a)?;
                float_un(b, *op, x)
            }
            Inst::Fma { a, b: rb, c, .. } => {
                // Scalar fused multiply-add — `a·b + c`, one rounding (matches the interp's `mul_add`).
                let (x, y, z) = (get(&vals, *a)?, get(&vals, *rb)?, get(&vals, *c)?);
                b.ins().fma(x, y, z)
            }
            Inst::FCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().fcmp(float_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Convert { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    ConvOp::ExtendI32S => b.ins().sextend(I64, x),
                    ConvOp::ExtendI32U => b.ins().uextend(I64, x),
                    ConvOp::WrapI64 => b.ins().ireduce(I32, x),
                }
            }
            Inst::Cast { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    CastOp::Demote => b.ins().fdemote(F32, x),
                    CastOp::Promote => b.ins().fpromote(F64, x),
                    CastOp::ReinterpI32F32 => b.ins().bitcast(F32, MemFlags::new(), x),
                    CastOp::ReinterpF32I32 => b.ins().bitcast(I32, MemFlags::new(), x),
                    CastOp::ReinterpI64F64 => b.ins().bitcast(F64, MemFlags::new(), x),
                    CastOp::ReinterpF64I64 => b.ins().bitcast(I64, MemFlags::new(), x),
                }
            }
            // Pointer ops (§3b/§10): plain 64-bit arithmetic off-CHERI — `ptr.add` is a
            // wrapping `iadd`, the int↔ptr casts pass the value through. Confinement is
            // untouched: these produce values; only `load`/`store` accesses are masked.
            Inst::PtrAdd { a, b: rb } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                b.ins().iadd(x, y)
            }
            Inst::PtrCast { a, .. } => get(&vals, *a)?,
            Inst::IToFConv { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let fty = float_clif_ty(to);
                if signed {
                    b.ins().fcvt_from_sint(fty, x)
                } else {
                    b.ins().fcvt_from_uint(fty, x)
                }
            }
            Inst::FToISat { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let ity = int_clif_ty(to);
                // Saturating (wasm trunc_sat): NaN→0, out-of-range→clamp — exactly
                // Cranelift's saturating fcvt, so it matches the interpreter.
                if signed {
                    b.ins().fcvt_to_sint_sat(ity, x)
                } else {
                    b.ins().fcvt_to_uint_sat(ity, x)
                }
            }
            Inst::FToITrap { op, a } => {
                let (from, to, signed) = op.parts();
                lower_trunc_trap(b, lower, get(&vals, *a)?, from, to, signed)
            }
            // ----- §17 SIMD (D58): native Cranelift vector lowering -----
            Inst::ConstV128(bytes) => {
                let c = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
                b.ins().vconst(I8X16, c)
            }
            Inst::V128Load { addr, offset, .. } => {
                // The 16-byte masked access — the lone escape-TCB delta SIMD adds (§17/D58).
                let elide = in_window(ub_at(&ubs, *addr), *offset, 16, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, 16);
                b.ins().load(I8X16, mem_flags(), phys, 0)
            }
            Inst::V128Store {
                addr,
                value,
                offset,
                ..
            } => {
                let elide = in_window(ub_at(&ubs, *addr), *offset, 16, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, 16);
                b.ins().store(mem_flags(), get(&vals, *value)?, phys, 0);
                continue; // store produces no value
            }
            Inst::Splat { shape, a } => {
                let s = get(&vals, *a)?;
                // The scalar arrives as the lane's `lane_val` (i32 for narrow ints); narrow to the
                // CLIF lane type, splat, then canonicalize to I8X16.
                let lane = lane_clif(*shape);
                let s = if b.func.dfg.value_type(s) == lane {
                    s
                } else if lane == I8 || lane == I16 {
                    b.ins().ireduce(lane, s)
                } else {
                    s
                };
                let v = b.ins().splat(vec_ty(*shape), s);
                vcast(b, v, I8X16)
            }
            Inst::ExtractLane {
                shape,
                lane,
                signed,
                a,
            } => {
                let v = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let raw = b.ins().extractlane(v, *lane);
                match shape {
                    // Narrow integer lanes widen to the i32 result (sign/zero per `signed`).
                    VShape::I8x16 | VShape::I16x8 => {
                        if *signed {
                            b.ins().sextend(I32, raw)
                        } else {
                            b.ins().uextend(I32, raw)
                        }
                    }
                    // i32x4/i64x2/f32x4/f64x2 extract to the lane type directly.
                    _ => raw,
                }
            }
            Inst::ReplaceLane {
                shape,
                lane,
                a,
                b: rb,
            } => {
                let v = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let s = get(&vals, *rb)?;
                let lty = lane_clif(*shape);
                let s = if b.func.dfg.value_type(s) == lty {
                    s
                } else if lty == I8 || lty == I16 {
                    b.ins().ireduce(lty, s)
                } else {
                    s
                };
                let r = b.ins().insertlane(v, s, *lane);
                vcast(b, r, I8X16)
            }
            Inst::VIntBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = match op {
                    VIntBinOp::Add => b.ins().iadd(x, y),
                    VIntBinOp::Sub => b.ins().isub(x, y),
                    // x86 has no per-byte multiply (no `PMULLB`), so Cranelift can't legalize an
                    // `imul` on `i8x16`. Emulate it: widen each half to `i16x8` and multiply (the low
                    // byte of an `i16` product equals the low byte of the `i8` product, and that low
                    // byte is sign-independent), mask each product to its low byte, then pack the two
                    // halves back with unsigned-saturating narrow (every lane is ≤ 0xFF so nothing
                    // saturates — it's an exact low-byte truncation). Matches the interp's wrapping mul.
                    VIntBinOp::Mul if *shape == VShape::I8x16 => {
                        let (xl, yl) = (b.ins().uwiden_low(x), b.ins().uwiden_low(y));
                        let (xh, yh) = (b.ins().uwiden_high(x), b.ins().uwiden_high(y));
                        let pl = b.ins().imul(xl, yl);
                        let ph = b.ins().imul(xh, yh);
                        let m = b.ins().iconst(I16, 0x00ff);
                        let mask = b.ins().splat(I16X8, m);
                        let pl = b.ins().band(pl, mask);
                        let ph = b.ins().band(ph, mask);
                        b.ins().unarrow(pl, ph)
                    }
                    VIntBinOp::Mul => b.ins().imul(x, y),
                    VIntBinOp::MinS => b.ins().smin(x, y),
                    VIntBinOp::MinU => b.ins().umin(x, y),
                    VIntBinOp::MaxS => b.ins().smax(x, y),
                    VIntBinOp::MaxU => b.ins().umax(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VIntCmp {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Vector `icmp` yields a per-lane all-ones/all-zeros mask of the lane width — exactly
                // the wasm/interp semantics — so this is a single instruction on the right vector type.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let cc = match op {
                    VICmpOp::Eq => IntCC::Equal,
                    VICmpOp::Ne => IntCC::NotEqual,
                    VICmpOp::LtS => IntCC::SignedLessThan,
                    VICmpOp::LtU => IntCC::UnsignedLessThan,
                    VICmpOp::GtS => IntCC::SignedGreaterThan,
                    VICmpOp::GtU => IntCC::UnsignedGreaterThan,
                    VICmpOp::LeS => IntCC::SignedLessThanOrEqual,
                    VICmpOp::LeU => IntCC::UnsignedLessThanOrEqual,
                    VICmpOp::GeS => IntCC::SignedGreaterThanOrEqual,
                    VICmpOp::GeU => IntCC::UnsignedGreaterThanOrEqual,
                };
                let r = b.ins().icmp(cc, x, y);
                vcast(b, r, I8X16)
            }
            Inst::VShift { shape, op, a, amt } => {
                // One scalar shift amount, masked to the lane bit-width (the wasm rule), broadcast
                // across the lanes by Cranelift's vector `ishl`/`ushr`/`sshr`.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let bits = (shape.lane_bytes() * 8) as i64;
                let sh = b.ins().band_imm(get(&vals, *amt)?, bits - 1);
                let r = match op {
                    VShiftOp::Shl => b.ins().ishl(x, sh),
                    VShiftOp::ShrU => b.ins().ushr(x, sh),
                    VShiftOp::ShrS => b.ins().sshr(x, sh),
                };
                vcast(b, r, I8X16)
            }
            Inst::VIntUn { shape, op, a } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let r = match op {
                    VIntUnOp::Abs => b.ins().iabs(x),
                    VIntUnOp::Neg => b.ins().ineg(x),
                };
                vcast(b, r, I8X16)
            }
            Inst::VPopcnt { a } => {
                // Canonical vectors are already I8X16, matching the op's fixed shape.
                b.ins().popcnt(get(&vals, *a)?)
            }
            Inst::VAvgr { shape, a, b: rb } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = b.ins().avg_round(x, y);
                vcast(b, r, I8X16)
            }
            Inst::VDot { a, b: rb } => {
                // Widen each i16x8 operand to two i32x4 halves, multiply lane-wise, then
                // horizontally add adjacent products: `iadd_pairwise([a0b0,a1b1,a2b2,a3b3],
                // [a4b4,..]) = [a0b0+a1b1, a2b2+a3b3, a4b4+a5b5, a6b6+a7b7]` — the wasm dot result.
                let x = vcast(b, get(&vals, *a)?, I16X8);
                let y = vcast(b, get(&vals, *rb)?, I16X8);
                let xl = b.ins().swiden_low(x);
                let xh = b.ins().swiden_high(x);
                let yl = b.ins().swiden_low(y);
                let yh = b.ins().swiden_high(y);
                let pl = b.ins().imul(xl, yl);
                let ph = b.ins().imul(xh, yh);
                let r = b.ins().iadd_pairwise(pl, ph);
                vcast(b, r, I8X16)
            }
            Inst::VDotI8 { a, b: rb } => {
                // The i8→i16 signed dot, same shape as `VDot` one width down: widen each i8x16 to
                // two i16x8 halves, multiply, pairwise-add. (Deterministic relaxed_dot_i8x16_i7x16_s.)
                let x = vcast(b, get(&vals, *a)?, I8X16);
                let y = vcast(b, get(&vals, *rb)?, I8X16);
                let xl = b.ins().swiden_low(x);
                let xh = b.ins().swiden_high(x);
                let yl = b.ins().swiden_low(y);
                let yh = b.ins().swiden_high(y);
                let pl = b.ins().imul(xl, yl);
                let ph = b.ins().imul(xh, yh);
                let r = b.ins().iadd_pairwise(pl, ph);
                vcast(b, r, I8X16)
            }
            Inst::VExtMul {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Widen the same (low/high, sign) half of both operands to the wide shape, then
                // multiply lane-wise — the wasm extmul.
                let (low, signed) = op.parts();
                let src = vec_ty(
                    shape
                        .narrower()
                        .expect("verifier ensures a narrower source"),
                );
                let x = vcast(b, get(&vals, *a)?, src);
                let y = vcast(b, get(&vals, *rb)?, src);
                let (wx, wy) = match (low, signed) {
                    (true, true) => (b.ins().swiden_low(x), b.ins().swiden_low(y)),
                    (false, true) => (b.ins().swiden_high(x), b.ins().swiden_high(y)),
                    (true, false) => (b.ins().uwiden_low(x), b.ins().uwiden_low(y)),
                    (false, false) => (b.ins().uwiden_high(x), b.ins().uwiden_high(y)),
                };
                let r = b.ins().imul(wx, wy);
                vcast(b, r, I8X16)
            }
            Inst::VExtAddPairwise { shape, signed, a } => {
                // Widen the low and high halves of the source, then pairwise-add: the two halves'
                // pairwise sums concatenate to `out[i] = w(a[2i]) + w(a[2i+1])`.
                let src = vec_ty(
                    shape
                        .narrower()
                        .expect("verifier ensures a narrower source"),
                );
                let x = vcast(b, get(&vals, *a)?, src);
                let (lo, hi) = if *signed {
                    (b.ins().swiden_low(x), b.ins().swiden_high(x))
                } else {
                    (b.ins().uwiden_low(x), b.ins().uwiden_high(x))
                };
                let r = b.ins().iadd_pairwise(lo, hi);
                vcast(b, r, I8X16)
            }
            Inst::VQ15MulrSat { a, b: rb } => {
                let x = vcast(b, get(&vals, *a)?, I16X8);
                let y = vcast(b, get(&vals, *rb)?, I16X8);
                let r = b.ins().sqmul_round_sat(x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFma {
                shape,
                neg,
                a,
                b: rb,
                c,
            } => {
                let ty = vec_ty(*shape);
                let xa = vcast(b, get(&vals, *a)?, ty);
                // `nmadd` is `−a·b + c`: negate the product by negating `a`.
                let x = if *neg { b.ins().fneg(xa) } else { xa };
                let y = vcast(b, get(&vals, *rb)?, ty);
                let z = vcast(b, get(&vals, *c)?, ty);
                let r = b.ins().fma(x, y, z);
                vcast(b, r, I8X16)
            }
            Inst::VSatBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = match op {
                    VSatBinOp::AddS => b.ins().sadd_sat(x, y),
                    VSatBinOp::AddU => b.ins().uadd_sat(x, y),
                    VSatBinOp::SubS => b.ins().ssub_sat(x, y),
                    VSatBinOp::SubU => b.ins().usub_sat(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VWiden { shape, op, a } => {
                // The source is the half-width shape; widen low/high → the wide result.
                let src_ty = vec_ty(shape.narrower().expect("verifier ensures a narrower shape"));
                let x = vcast(b, get(&vals, *a)?, src_ty);
                let (low, signed) = op.parts();
                let r = match (low, signed) {
                    (true, true) => b.ins().swiden_low(x),
                    (false, true) => b.ins().swiden_high(x),
                    (true, false) => b.ins().uwiden_low(x),
                    (false, false) => b.ins().uwiden_high(x),
                };
                vcast(b, r, I8X16)
            }
            Inst::VNarrow {
                shape,
                op,
                a,
                b: rb,
            } => {
                // The sources are the wider shape; `snarrow`/`unarrow` saturate `a`'s lanes then
                // `b`'s into the narrow result.
                let src_ty = vec_ty(shape.wider().expect("verifier ensures a wider source"));
                let x = vcast(b, get(&vals, *a)?, src_ty);
                let y = vcast(b, get(&vals, *rb)?, src_ty);
                let r = match op {
                    VNarrowOp::S => b.ins().snarrow(x, y),
                    VNarrowOp::U => b.ins().unarrow(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VConvert { op, a } => {
                let raw = get(&vals, *a)?;
                let r = match op {
                    VCvtOp::F32x4ConvertI32x4S => {
                        let x = vcast(b, raw, I32X4);
                        b.ins().fcvt_from_sint(F32X4, x)
                    }
                    VCvtOp::F32x4ConvertI32x4U => {
                        let x = vcast(b, raw, I32X4);
                        b.ins().fcvt_from_uint(F32X4, x)
                    }
                    VCvtOp::I32x4TruncSatF32x4S => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fcvt_to_sint_sat(I32X4, x)
                    }
                    VCvtOp::I32x4TruncSatF32x4U => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fcvt_to_uint_sat(I32X4, x)
                    }
                    VCvtOp::F32x4DemoteF64x2Zero => {
                        let x = vcast(b, raw, F64X2);
                        b.ins().fvdemote(x)
                    }
                    VCvtOp::F64x2PromoteLowF32x4 => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fvpromote_low(x)
                    }
                    // Lane-count changes (2↔4). Widen/narrow through the i64x2 intermediate, the
                    // same recipe Cranelift's own wasm frontend uses.
                    VCvtOp::F64x2ConvertLowI32x4S => {
                        let x = vcast(b, raw, I32X4);
                        let w = b.ins().swiden_low(x); // low 2 i32 → i64x2
                        b.ins().fcvt_from_sint(F64X2, w)
                    }
                    VCvtOp::F64x2ConvertLowI32x4U => {
                        let x = vcast(b, raw, I32X4);
                        let w = b.ins().uwiden_low(x); // low 2 u32 → i64x2
                        b.ins().fcvt_from_uint(F64X2, w)
                    }
                    VCvtOp::I32x4TruncSatF64x2SZero => {
                        let x = vcast(b, raw, F64X2);
                        let conv = b.ins().fcvt_to_sint_sat(I64X2, x); // i64x2
                        let zc = b
                            .func
                            .dfg
                            .constants
                            .insert(ConstantData::from(&[0u8; 16][..]));
                        let zero = b.ins().vconst(I64X2, zc);
                        // snarrow packs [conv | zero] → i32x4: low 2 lanes = conv, high 2 = 0.
                        b.ins().snarrow(conv, zero)
                    }
                    VCvtOp::I32x4TruncSatF64x2UZero => {
                        let x = vcast(b, raw, F64X2);
                        let conv = b.ins().fcvt_to_uint_sat(I64X2, x); // i64x2
                        let zc = b
                            .func
                            .dfg
                            .constants
                            .insert(ConstantData::from(&[0u8; 16][..]));
                        let zero = b.ins().vconst(I64X2, zc);
                        b.ins().uunarrow(conv, zero)
                    }
                };
                vcast(b, r, I8X16)
            }
            // Boolean reductions → an `i32`. `vany_true`/`vall_true` yield an `I8` bool (zero/one),
            // widened to `i32`; `vhigh_bits` produces the bitmask directly into an `i32`.
            Inst::VAnyTrue { a } => {
                let x = get(&vals, *a)?; // shape-agnostic; the canonical I8X16 view is fine
                let t = b.ins().vany_true(x);
                b.ins().uextend(I32, t)
            }
            Inst::VAllTrue { shape, a } => {
                let x = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let t = b.ins().vall_true(x);
                b.ins().uextend(I32, t)
            }
            Inst::VBitmask { shape, a } => {
                let x = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                b.ins().vhigh_bits(I32, x)
            }
            Inst::VFloatCmp {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Vector `fcmp` yields a per-lane all-ones/all-zeros mask — the wasm/interp semantics.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let cc = match op {
                    VFCmpOp::Eq => FloatCC::Equal,
                    VFCmpOp::Ne => FloatCC::NotEqual,
                    VFCmpOp::Lt => FloatCC::LessThan,
                    VFCmpOp::Gt => FloatCC::GreaterThan,
                    VFCmpOp::Le => FloatCC::LessThanOrEqual,
                    VFCmpOp::Ge => FloatCC::GreaterThanOrEqual,
                };
                let r = b.ins().fcmp(cc, x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFloatBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                // Reuse the scalar float lowering — Cranelift's `fadd`/`fmin`/… are polymorphic over
                // scalar and vector, so lanes lower the same way scalars do.
                let r = float_bin(b, vf_bin(*op), x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFloatUn { shape, op, a } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let r = float_un(b, vf_un(*op), x);
                vcast(b, r, I8X16)
            }
            Inst::VPMinMax {
                shape,
                op,
                a,
                b: rb,
            } => {
                // pmin(a,b) = b < a ? b : a ; pmax(a,b) = a < b ? b : a.
                // Both select the second operand `b` where a one-sided `<` holds — only the
                // compare operand order differs. `fcmp` yields the lane mask (`I32X4`/`I64X2`),
                // which we blend in the canonical `I8X16` domain so `bitselect`'s three args
                // share a type. This matches the interp's NaN/-0 propagation (no IEEE min/max).
                let ty = vec_ty(*shape);
                let xc = get(&vals, *a)?;
                let yc = get(&vals, *rb)?;
                let x = vcast(b, xc, ty);
                let y = vcast(b, yc, ty);
                let mask = match op {
                    VPMinMaxOp::Pmin => b.ins().fcmp(FloatCC::LessThan, y, x),
                    VPMinMaxOp::Pmax => b.ins().fcmp(FloatCC::LessThan, x, y),
                };
                let m = vcast(b, mask, I8X16);
                b.ins().bitselect(m, yc, xc)
            }
            Inst::VBitBin { op, a, b: rb } => {
                // Whole-vector — operate on the canonical I8X16 directly.
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                match op {
                    VBitBinOp::And => b.ins().band(x, y),
                    VBitBinOp::Or => b.ins().bor(x, y),
                    VBitBinOp::Xor => b.ins().bxor(x, y),
                    VBitBinOp::AndNot => b.ins().band_not(x, y),
                }
            }
            Inst::VNot { a } => b.ins().bnot(get(&vals, *a)?),
            Inst::Bitselect { a, b: rb, mask } => {
                // IR `(a & mask) | (b & !mask)` == Cranelift `bitselect(mask, a, b)`.
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                let m = get(&vals, *mask)?;
                b.ins().bitselect(m, x, y)
            }
            Inst::Shuffle { lanes, a, b: rb } => {
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                let imm = b.func.dfg.immediates.push(ConstantData::from(&lanes[..]));
                b.ins().shuffle(x, y, imm)
            }
            Inst::Swizzle { a, b: rb } => {
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                b.ins().swizzle(x, y)
            }
            // §17/D58 feature-detect hook: the fixed-128 constant (matches the interpreter).
            Inst::SimdWidthBytes => b.ins().iconst(I32, 16),

            Inst::Load {
                op, addr, offset, ..
            } => {
                let elide = in_window(ub_at(&ubs, *addr), *offset, op.info().2, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, op.info().2);
                lower_load(b, *op, phys)
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let elide = in_window(ub_at(&ubs, *addr), *offset, op.info().2, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, op.info().2);
                lower_store(b, *op, phys, get(&vals, *value)?);
                continue; // store produces no value
            }
            // Bulk-memory ops (D62). Each span is confined once (single range check + base clamp),
            // then the copy/fill is the platform libcall — `memory.copy`-class lowering, not a
            // per-byte confined loop. `MemCopy`/`MemMove` differ only in the libcall (overlap safety).
            Inst::MemCopy { dst, src, len } => {
                let n = get(&vals, *len)?;
                let dphys = confine_span(b, lower, get(&vals, *dst)?, n);
                let sphys = confine_span(b, lower, get(&vals, *src)?, n);
                // I21: fault an overrun before the libcall (no partial write; catches `dst==src`).
                probe_span(b, dphys, n);
                probe_span(b, sphys, n);
                b.call_memcpy(lower.frontend_config, dphys, sphys, n);
                continue;
            }
            Inst::MemMove { dst, src, len } => {
                let n = get(&vals, *len)?;
                let dphys = confine_span(b, lower, get(&vals, *dst)?, n);
                let sphys = confine_span(b, lower, get(&vals, *src)?, n);
                probe_span(b, dphys, n);
                probe_span(b, sphys, n);
                b.call_memmove(lower.frontend_config, dphys, sphys, n);
                continue;
            }
            Inst::MemFill { dst, val, len } => {
                let n = get(&vals, *len)?;
                let dphys = confine_span(b, lower, get(&vals, *dst)?, n);
                probe_span(b, dphys, n);
                // `call_memset` uextends the fill byte to i32, so hand it the low byte (i8).
                let v8 = b.ins().ireduce(I8, get(&vals, *val)?);
                b.call_memset(lower.frontend_config, dphys, v8, n);
                continue;
            }
            // §12 atomics. Confine like a normal access, then a natural-alignment guard (a misaligned
            // address traps — `atomic_*` require alignment, and it matches the interpreter), then a
            // hardware atomic. Elision uses the same upper-bound analysis.
            // The `order` is ignored: Cranelift atomics are seq-cst, which soundly implements every
            // requested ordering and keeps the interpreter↔JIT oracle exact (see `svm_ir::Ordering`).
            Inst::AtomicLoad {
                ty, addr, offset, ..
            } => {
                let w = atomic_width(*ty);
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                b.ins().atomic_load(int_clif_ty(*ty), atomic_flags(), phys)
            }
            Inst::AtomicStore {
                ty,
                addr,
                value,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                b.ins()
                    .atomic_store(atomic_flags(), get(&vals, *value)?, phys);
                continue; // atomic store produces no value
            }
            Inst::AtomicRmw {
                ty,
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                b.ins().atomic_rmw(
                    int_clif_ty(*ty),
                    atomic_flags(),
                    clif_rmw_op(*op),
                    phys,
                    get(&vals, *value)?,
                )
            }
            Inst::AtomicCmpxchg {
                ty,
                addr,
                expected,
                replacement,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w); // type is inferred from the operands
                b.ins().atomic_cas(
                    atomic_flags(),
                    phys,
                    get(&vals, *expected)?,
                    get(&vals, *replacement)?,
                )
            }
            // A funcref is just the function index as plain i32 data (§3c) — the same
            // value the interpreter materializes; `call_indirect` masks it into the table.
            Inst::RefFunc { func } => b.ins().iconst(I32, *func as i64),
            // §12 standalone fence. Cranelift emits a full (seq-cst) barrier regardless of the
            // requested `order` — the same sound strengthening the atomics use.
            Inst::AtomicFence { .. } => {
                b.ins().fence();
                continue; // produces no value
            }
            _ => return Err(JitError::Unsupported("instruction")),
        };
        // Single-result instruction: record its value and a sound upper bound in lockstep.
        let u = ub_of(inst, &ubs);
        vals.push(v);
        ubs.push(u);
    }

    // W5 JIT/DWARF Stage 3a: now that this block's value map is fully populated, stamp the CLIF
    // values backing source variables with their `ValueLabel`, so Cranelift records the
    // machine-location ranges (`value_labels_ranges`) `compile` reads back. `set_val_label` is inert
    // unless `collect_debug_info` was enabled (it is, exactly when `var_labels` is `Some`), so
    // non-`-g` codegen is untouched. The label association drives liveness-based range computation;
    // the block-local `value` index maps straight onto `vals`.
    if let Some(map) = lower.var_labels {
        if let Some(points) = map.get(&(lower.func_idx, block_idx as u32)) {
            for &(value, label) in points {
                if let Some(&v) = vals.get(value as usize) {
                    b.set_val_label(v, ValueLabel::from_u32(label));
                }
            }
        }
    }

    match &blk.term {
        Terminator::Br { target, args } => {
            let ba = map_args(&vals, args)?;
            let t = *blocks.get(*target as usize).ok_or(JitError::Malformed)?;
            // §5 kill-path: poll the interrupt cell before taking any branch — every loop body ends
            // in one of these terminators, so this bounds a non-terminating intra-function loop.
            emit_epoch_check(b, lower);
            b.ins().jump(t, &ba);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let c = get(&vals, *cond)?;
            let ta = map_args(&vals, then_args)?;
            let ea = map_args(&vals, else_args)?;
            let tb = *blocks.get(*then_blk as usize).ok_or(JitError::Malformed)?;
            let eb = *blocks.get(*else_blk as usize).ok_or(JitError::Malformed)?;
            emit_epoch_check(b, lower); // §5 kill-path (see `Br`)
            b.ins().brif(c, tb, &ta, eb, &ea);
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let index = get(&vals, *idx)?;
            emit_epoch_check(b, lower); // §5 kill-path (see `Br`)
                                        // Build a BlockCall (target block + its edge args) for each table entry
                                        // and the default; Cranelift masks the index and selects, default on OOB.
            let mut entries = Vec::with_capacity(targets.len());
            for (t, args) in targets {
                let ba = map_args(&vals, args)?;
                let blk = *blocks.get(*t as usize).ok_or(JitError::Malformed)?;
                entries.push(BlockCall::new(
                    blk,
                    ba.iter().copied(),
                    &mut b.func.dfg.value_lists,
                ));
            }
            let (dt, dargs) = default;
            let dba = map_args(&vals, dargs)?;
            let dblk = *blocks.get(*dt as usize).ok_or(JitError::Malformed)?;
            let dcall = BlockCall::new(dblk, dba.iter().copied(), &mut b.func.dfg.value_lists);
            let jt = b.create_jump_table(JumpTableData::new(dcall, &entries));
            b.ins().br_table(index, jt);
        }
        Terminator::Return(outs) => {
            let rets: Vec<Value> = outs
                .iter()
                .map(|o| get(&vals, *o))
                .collect::<Result<_, _>>()?;
            if let Some(sret_var) = lower.sret_var {
                // sret ABI: write each result into the caller-provided return-area (8-byte slots,
                // `encode_slot`-encoded like the buffer ABI), then return void.
                let ptr = b.use_var(sret_var);
                for (i, r) in rets.iter().enumerate() {
                    let slot = encode_slot(b, *r);
                    b.ins()
                        .store(MemFlags::trusted(), slot, ptr, (i * 8) as i32);
                }
                b.ins().return_(&[]);
            } else {
                // Natural ABI: return the result values directly (CLIF multi-return).
                b.ins().return_(&rets);
            }
        }
        Terminator::ReturnCall { func, args } => {
            // Tail call (§3b): replace this frame with the callee, threading the context.
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = ctx_args(b, lower);
            // The tail callee shares this function's result type (verifier-enforced), so its sret-ness
            // matches ours: when we return via sret, forward our own return-area pointer.
            if let Some(sret_var) = lower.sret_var {
                cargs.push(b.use_var(sret_var));
            }
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            b.ins().return_call(callee, &cargs);
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            // Indirect tail call: table dispatch (§3c) then a guaranteed tail call.
            let code = indirect_dispatch(b, lower, get(&vals, *idx)?, ty);
            let sig = b.import_signature(sig_from(module, &ty.params, &ty.results));
            let mut cargs = ctx_args(b, lower);
            // `ty.results` equals this function's results (tail-call contract), so sret-ness matches:
            // forward our return-area pointer when we return via sret.
            if let Some(sret_var) = lower.sret_var {
                cargs.push(b.use_var(sret_var));
            }
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            b.ins().return_call_indirect(sig, code, &cargs);
        }
        Terminator::Unreachable => {
            emit_trap(b, lower, TrapKind::Unreachable);
        }
    }
    Ok(())
}

/// The leading context arguments threaded into every guest call: `(mem_base,
/// fn_table_base, trap_out, stack_limit)`.
fn ctx_args(b: &mut FunctionBuilder, lower: &Lower) -> Vec<Value> {
    vec![
        b.use_var(lower.mem_var),
        b.use_var(lower.fn_table_var),
        b.use_var(lower.trap_var),
        // §2b path B: pass our own stack-limit on to the callee (same stack, same limit) — mirrors
        // sig_from. Constant within a stack's call tree; set anew at each fiber/root entry.
        b.use_var(lower.limit_var),
    ]
}

/// For a callee whose `results` use the sret ABI ([`uses_sret`]), allocate a stack **return-area**,
/// push its address onto `cargs` (right after the context pointers, before the user args — the order
/// [`sig_from`] lays out), and return the slot; else push nothing and return `None`.
fn sret_call_slot(
    b: &mut FunctionBuilder,
    cargs: &mut Vec<Value>,
    results: &[ValType],
) -> Option<cranelift_codegen::ir::StackSlot> {
    if !uses_sret(results) {
        return None;
    }
    let ss = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (results.len() * 8) as u32,
        3, // 8-byte aligned slots
    ));
    let addr = b.ins().stack_addr(I64, ss, 0);
    cargs.push(addr);
    Some(ss)
}

/// Append a call's results to `vals`: from registers ([`FunctionBuilder::inst_results`]) on the
/// normal path, or by loading the sret return-area's 8-byte slots ([`decode_slot`]-encoded, matching
/// the storing `Return`) on the sret path.
fn read_call_results(
    b: &mut FunctionBuilder,
    call: cranelift_codegen::ir::Inst,
    sret: Option<cranelift_codegen::ir::StackSlot>,
    results: &[ValType],
    vals: &mut Vec<Value>,
) {
    match sret {
        None => vals.extend_from_slice(b.inst_results(call)),
        Some(ss) => {
            for (i, r) in results.iter().enumerate() {
                let raw = b.ins().stack_load(I64, ss, (i * 8) as i32);
                vals.push(decode_slot(b, raw, *r));
            }
        }
    }
}

/// Lower a trap (§5 detect-and-kill): store the kind code into the host trap cell, then
/// `return` dummy zero results so the run unwinds to the trampoline, which reports the
/// trap. (The reference JIT detects traps this way; production uses hardware faults.)
///
/// Caveat: this returns from the *current* function only. The current scalar tests put
/// every trap in the entry function (or its dispatch), so that suffices; propagating a
/// trap *out of a callee* would need a post-call check, added when a case needs it.
fn emit_trap(b: &mut FunctionBuilder, lower: &Lower, kind: TrapKind) {
    emit_trap_set(b, lower, kind);
    emit_trap_return(b, lower);
}

/// Record a trap in the cell (with the §5 W3 backtrace capture) **without** terminating the block.
/// Callers pair it with [`emit_trap_return`] (an unconditional trap, via [`emit_trap`]) or with
/// [`emit_trap_propagate`] (the shared branch-on-cell exit) when compilation of the rest of the
/// block must continue — the branch keeps the continuation formally reachable, so later
/// instructions still lower against a well-formed value stack even though they are dead at runtime.
fn emit_trap_set(b: &mut FunctionBuilder, lower: &Lower, kind: TrapKind) {
    // §5 W3 Stage 2: record a source backtrace for this explicit trap *before* it unwinds — the trap
    // stores its kind and returns, and that return propagates up tearing down every guest frame, so
    // the helper must walk the frame-pointer chain from this live site now. The current op's
    // `SourceLoc` is in effect, so the call inherits it and symbolizes to the trapping op's line.
    // Only present under `-g` (else the backtrace would be empty); the trap path is otherwise
    // unchanged.
    if let Some((sigref, addr)) = lower.trap_capture {
        // Pass the trapping function's frame pointer so the helper can walk the caller chain without
        // `__builtin_frame_address` (which MSVC lacks); it pairs `fp` with its own return address (the
        // trap site) for the innermost frame.
        let fp = b.ins().get_frame_pointer(I64);
        let helper = b.ins().iconst(I64, addr);
        b.ins().call_indirect(sigref, helper, &[fp]);
    }
    let cell = b.use_var(lower.trap_var);
    let code = b.ins().iconst(I64, kind as u32 as i64); // full i64 cell (high bits 0)
    b.ins().store(MemFlags::trusted(), code, cell, 0);
}

/// Emit the function `return` on a trap / early-exit path: **void** for an sret function (its results
/// flow through the return-area pointer, left unwritten — the trap cell is set, so the caller returns
/// before reading them), else dummy zeros of the result types (the register-ABI convention).
fn emit_trap_return(b: &mut FunctionBuilder, lower: &Lower) {
    if lower.sret_var.is_some() {
        b.ins().return_(&[]);
    } else {
        let zeros: Vec<Value> = lower.result_tys.iter().map(|t| zero_of(b, *t)).collect();
        b.ins().return_(&zeros);
    }
}

/// Emit the §5 fuel/epoch **kill-path** check: if the host has set the interrupt cell non-zero
/// (a watchdog timer, a cross-domain preemption, …), trap [`TrapKind::OutOfFuel`]; otherwise fall
/// through. A no-op when no kill-path is armed (`epoch_addr == 0`) — then the guest code is emitted
/// exactly as before. Placed at function entry and every loop back-edge, so any non-terminating
/// guest polls the cell within a bounded number of steps and is stopped — the guest can't disable
/// the poll (only the host writes the cell). The load is **not** `readonly`, so it is re-evaluated
/// each iteration (never hoisted out of the loop); it never faults (a host-owned aligned cell).
///
/// On return the builder is positioned at a fresh continuation block (the not-interrupted path);
/// the caller emits the real terminator / jump there.
fn emit_epoch_check(b: &mut FunctionBuilder, lower: &Lower) {
    if lower.epoch_addr == 0 {
        return; // no kill-path armed for this compile — emit nothing
    }
    let cont = b.create_block();
    let trap_blk = b.create_block();
    let addr = b.ins().iconst(I64, lower.epoch_addr);
    // The host stores into this cell **concurrently** (the watchdog thread), so the poll must reload
    // it every check. An **atomic** load is the reliable way to say so: under `opt_level=speed`
    // Cranelift's alias analysis sees no *guest* store to the cell and would hoist/CSE a plain load out
    // of the loop (the poll would fire once and never again — the runaway would never be killed); an
    // atomic load is a synchronization op the optimizer won't hoist. (The cell is a host `AtomicU64`.)
    let flag = b.ins().atomic_load(I64, atomic_flags(), addr);
    b.ins().brif(flag, trap_blk, &[], cont, &[]);
    b.switch_to_block(trap_blk);
    emit_trap(b, lower, TrapKind::OutOfFuel);
    b.switch_to_block(cont);
    // `cont`/`trap_blk` are sealed by the caller's `seal_all_blocks`.
}

/// The software stack-overflow guard's tunables. See `crates/svm-jit/STACK_GUARD.md`. Under §2b
/// **path B** the per-thread limit is NOT a global cell: it is a value ABI param (`stack_limit`,
/// [`Lower::limit_var`]) threaded through every call and set anew at each fiber entry from
/// `Yielder::stack_low()` — per-vCPU by construction, no cell and no TLS. The prologue check
/// ([`emit_stack_check`]) reads it straight from that register. Always compiled: the guard is in the
/// always-on escape-TCB path (it becomes the sole overflow defense once the arena drops the guard
/// page, so it must not be feature-gated out).
pub(crate) mod stack_check {
    /// Headroom (bytes) the prologue check reserves below the limit. Because the check runs *after* the
    /// machine prologue's `sub rsp` (it's in the entry block), it validates this function's frame
    /// directly; `RED_ZONE` only needs to cover the prologue's pre-check register pushes plus the
    /// check's own scratch. A single frame larger than the whole stack is a residual backstop concern
    /// (STACK_GUARD.md §2b) — Cranelift 0.132 doesn't expose the final frame size to enforce it.
    pub(crate) const RED_ZONE: u64 = 1 << 14; // 16 KiB
}

/// The per-prologue software stack-limit check — overflow protection for the arena/software-guard
/// fiber model, which drops the per-fiber hardware guard page. Always emitted (escape-TCB). Reads the
/// running fiber's low bound from our own `stack_limit` ABI param ([`Lower::limit_var`], §2b path B —
/// per-vCPU by construction, no cell and no TLS), gets the current SP, and traps
/// [`TrapKind::StackOverflow`] if `SP - RED_ZONE < limit` — i.e. this function's frame would grow the
/// native stack past the fiber's low bound. The check sits in the entry block, *after* the machine
/// prologue's `sub rsp`, so `get_stack_pointer` already reflects this frame and it is validated
/// directly (soundness relies on Cranelift not page-probing during `sub rsp` — `enable_probestack` is
/// off; see the ISA flags). `limit == 0` (the root / spawned-vCPU top, on OS-guarded stacks) ⇒
/// `SP - RED_ZONE` is a huge address, never unsigned-`< 0`, so the check is inert there. The limit is a
/// constant within a stack's call tree, so the callee re-checks before its own frame.
fn emit_stack_check(b: &mut FunctionBuilder, lower: &Lower) {
    let cont = b.create_block();
    let trap_blk = b.create_block();
    // §2b path B: the running stack's low bound, from our own ABI param (per-vCPU by construction — no
    // cell, no TLS). 0 for the root/thread-top ⇒ `SP - RED_ZONE < 0` is never true ⇒ inert (OS guard).
    let limit = b.use_var(lower.limit_var);
    let sp = b.ins().get_stack_pointer(I64);
    let guard = b.ins().iadd_imm(sp, -(stack_check::RED_ZONE as i64)); // SP - RED_ZONE
    let lt = b.ins().icmp(IntCC::UnsignedLessThan, guard, limit);
    b.ins().brif(lt, trap_blk, &[], cont, &[]);
    b.switch_to_block(trap_blk);
    emit_trap(b, lower, TrapKind::StackOverflow);
    b.switch_to_block(cont);
}

/// A zero constant of CLIF type `t` (for a trapping path's dummy return).
fn zero_of(b: &mut FunctionBuilder, t: Type) -> Value {
    if t == F32 {
        b.ins().f32const(0.0)
    } else if t == F64 {
        b.ins().f64const(0.0)
    } else if t.is_vector() {
        // A vector result (e.g. a `float4`/v128 returned in a register, per §17): `iconst` is
        // scalar-integer only and produces malformed IR (the verifier hits an internal `unreachable`),
        // so materialize an all-zero vector constant instead — the same idiom the SIMD lowering uses.
        let bytes = vec![0u8; t.bytes() as usize];
        let zc = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
        b.ins().vconst(t, zc)
    } else {
        b.ins().iconst(t, 0)
    }
}

/// The §3c indirect-call dispatch (invariant I2): mask the guest index into the
/// host-owned function table, load the slot's `type_id`, trap if it does not match the
/// call's signature (a forged/wrong-type index is inert), and return the slot's code
/// pointer. Leaves the builder positioned in the post-check ("matched") block.
fn indirect_dispatch(b: &mut FunctionBuilder, lower: &Lower, idx: Value, ty: &FuncType) -> Value {
    // slot = (idx as u32) & (next_pow2(nfuncs) - 1) — mask, not branch (Spectre-safe).
    let idx64 = b.ins().uextend(I64, idx);
    let m = b.ins().iconst(I64, lower.fn_table_mask as i64);
    let slot = b.ins().band(idx64, m);
    // entry_addr = fn_table_base + slot * sizeof(FnEntry=16)
    let off = b.ins().imul_imm(slot, 16);
    let base = b.use_var(lower.fn_table_var);
    let entry_addr = b.ins().iadd(base, off);

    // type_id check against the call's expected id.
    let tid = b.ins().load(I32, MemFlags::trusted(), entry_addr, 0);
    let expected = type_id_of(lower.distinct, ty);
    let want = b.ins().iconst(I32, expected as i32 as i64);
    let cond = b.ins().icmp(IntCC::Equal, tid, want);
    let matched = b.create_block();
    let bad = b.create_block();
    b.ins().brif(cond, matched, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::IndirectCallType);
    b.switch_to_block(matched);
    b.seal_block(matched);
    // code pointer at offset 8.
    b.ins().load(I64, MemFlags::trusted(), entry_addr, 8)
}

/// Emit "if the host trap cell is non-zero, propagate the trap now": branch to an early
/// `return` of zero-valued results (the trap kind / exit code already sits in the cell, which the
/// entry trampoline reads to decide `Trapped`/`Exited`). Used after **every** `cap.call` *and*
/// every `call`/`call_indirect`, so a trap raised deep in a callee unwinds the whole guest stack
/// immediately — before any later op can observe bogus zero results or overwrite the cell (a
/// *successful* `cap.call` resets it to 0, which would otherwise mask a callee's trap).
fn emit_trap_propagate(b: &mut FunctionBuilder, lower: &Lower) {
    let trap_out = b.use_var(lower.trap_var);
    let tc = b.ins().load(I64, MemFlags::trusted(), trap_out, 0);
    let trapped = b.ins().icmp_imm(IntCC::NotEqual, tc, 0);
    let trapret = b.create_block();
    let cont = b.create_block();
    b.ins().brif(trapped, trapret, &[], cont, &[]);
    b.switch_to_block(trapret);
    b.seal_block(trapret);
    emit_trap_return(b, lower);
    b.switch_to_block(cont);
    b.seal_block(cont);
}

/// Resolve the §9/D45 fast-path target for a `cap.call`, if one applies: there is a [`FastCapResolver`]
/// (top-level compile), the op has **at most one result** (the register ABI returns a single i64), and
/// the resolver claims this `(type_id, op)` (returns a non-null specialized fn). Returns the baked
/// target address. Resolution is a **compile-time** call into the embedder's resolver.
fn fast_cap_target(lower: &Lower, type_id: u32, op: u32, sig: &FuncType) -> Option<i64> {
    let resolver = lower.cap.fast_resolver?;
    if sig.results.len() > 1 {
        return None; // the fast ABI returns a single register; multi-result falls back to the thunk
    }
    // SAFETY: `resolver` honours the `FastCapResolver` contract (caller guarantee); it's a pure
    // `(type_id, op, n_args, n_res) -> *const fn` lookup with no side effects, safe to call during
    // codegen. The arity is passed so the resolver only claims an op when its specialized fn's arity
    // matches the IR `cap.call`'s — else it returns null and the generic slot-based path is used.
    let target = unsafe {
        resolver(
            type_id,
            op,
            sig.params.len() as u32,
            sig.results.len() as u32,
        )
    } as i64;
    (target != 0).then_some(target)
}

/// Lower a `cap.call` via the **devirtualized fast path** (§9 / D45): a direct register-to-register
/// call to the specialized host fn at `target`, passing `ctx`/`mem_base`/`mem_size`/`handle`/`trap_out`
/// then each argument **in a register** (widened to its i64 slot), and reading the single result back
/// from a register — no stack-slot marshalling, no `n_args`/`n_res`/`type_id`/`op` dispatch. The trap
/// cell is checked exactly as in [`lower_cap_call`].
#[allow(clippy::too_many_arguments)]
fn lower_cap_call_fast(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    target: i64,
    sig: &FuncType,
    handle: u32,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    let n_res = sig.results.len();
    let ctx = b.ins().iconst(I64, lower.cap.ctx_addr);
    let mem_base = b.use_var(lower.mem_var);
    let mem_size = b.ins().iconst(I64, lower.mapped as i64);
    let h = get(vals, handle)?;
    let trap_out = b.use_var(lower.trap_var);

    // Signature: (ctx, mem_base, mem_size, handle:i32, trap_out, args…:i64) -> [i64].
    let mut tsig = module.make_signature();
    for t in [I64, I64, I64, I32, I64] {
        tsig.params.push(AbiParam::new(t));
    }
    for _ in args {
        tsig.params.push(AbiParam::new(I64));
    }
    if n_res == 1 {
        tsig.returns.push(AbiParam::new(I64));
    }
    let tsigref = b.import_signature(tsig);

    let mut call_args = vec![ctx, mem_base, mem_size, h, trap_out];
    for a in args {
        let v = get(vals, *a)?;
        call_args.push(encode_slot(b, v)); // each arg widened to its i64 register slot
    }
    let target_v = b.ins().iconst(I64, target);
    let call = b.ins().call_indirect(tsigref, target_v, &call_args);

    // If the specialized fn set the trap cell, propagate (the cell already holds the kind / exit code).
    emit_trap_propagate(b, lower);

    if n_res == 1 {
        let slot = b.inst_results(call)[0]; // the i64 return register
        vals.push(decode_slot(b, slot, sig.results[0]));
    }
    Ok(())
}

/// Lower a `cap.call` (§3c/§9): marshal the arg slots into a stack buffer, call the host
/// thunk (a baked-in constant address) with the cap immediates + the guest window, and
/// — unless it set the trap cell — read the result slots back. A trap from the thunk
/// (CapFault / Exit) propagates like any other (return early; the cell is already set).
#[allow(clippy::too_many_arguments)]
fn lower_cap_call(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    type_id: u32,
    op: u32,
    sig: &FuncType,
    // The already-resolved `i32` handle value (the caller does `get(vals, handle_idx)`; §7 reflection
    // passes a constant 0 — the `CAP_SELF_TYPE_ID` dispatch ignores it).
    h: Value,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    let n_args = args.len();
    let n_res = sig.results.len();

    // Marshal the args into a stack buffer of i64 slots (null pointer when there are 0).
    let args_ptr = if n_args == 0 {
        b.ins().iconst(I64, 0)
    } else {
        let ss = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (n_args * 8) as u32,
            3,
        ));
        let addr = b.ins().stack_addr(I64, ss, 0);
        for (i, a) in args.iter().enumerate() {
            let v = get(vals, *a)?;
            let slot = encode_slot(b, v);
            b.ins()
                .store(MemFlags::trusted(), slot, addr, (i * 8) as i32);
        }
        addr
    };
    let res_ss = if n_res == 0 {
        None
    } else {
        Some(b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (n_res * 8) as u32,
            3,
        )))
    };
    let res_ptr = match res_ss {
        Some(ss) => b.ins().stack_addr(I64, ss, 0),
        None => b.ins().iconst(I64, 0),
    };
    // Audit #4: pre-zero the result slots, so a host op that writes fewer than `n_res` results can't
    // leave uninitialized stack for the read-back to decode. The verifier pins the sig arity, so a
    // correct host fills all of them — this guards a buggy host, not a guest (bounded to this slot).
    if let Some(ss) = res_ss {
        let z = b.ins().iconst(I64, 0);
        for i in 0..n_res {
            b.ins().stack_store(z, ss, (i * 8) as i32);
        }
    }

    // Assemble the thunk arguments (see `CapThunk`).
    let ctx = b.ins().iconst(I64, lower.cap.ctx_addr);
    let mem_base = b.use_var(lower.mem_var);
    let mem_size = b.ins().iconst(I64, lower.mapped as i64);
    // The reserved mask domain (`mask + 1`) the guest may `map`-grow into; 0 when no memory.
    let reserved = if lower.mapped == 0 { 0 } else { lower.mask + 1 };
    let mem_reserved = b.ins().iconst(I64, reserved as i64);
    let tid = b.ins().iconst(I32, type_id as i64);
    let opc = b.ins().iconst(I32, op as i64);
    let na = b.ins().iconst(I64, n_args as i64);
    let nr = b.ins().iconst(I64, n_res as i64);
    let trap_out = b.use_var(lower.trap_var);
    let thunk = b.ins().iconst(I64, lower.cap.thunk_addr);

    let mut tsig = module.make_signature(); // host C ABI (matches `extern "C"`)
    for t in [I64, I64, I64, I64, I32, I32, I32, I64, I64, I64, I64, I64] {
        tsig.params.push(AbiParam::new(t));
    }
    let tsigref = b.import_signature(tsig);
    let call_args = [
        ctx,
        mem_base,
        mem_size,
        mem_reserved,
        tid,
        opc,
        h,
        args_ptr,
        na,
        res_ptr,
        nr,
        trap_out,
    ];
    b.ins().call_indirect(tsigref, thunk, &call_args);

    // If the thunk set the trap cell, propagate (return early; the cell already holds
    // the kind / exit code).
    emit_trap_propagate(b, lower);

    // Read the result slots back.
    if let Some(ss) = res_ss {
        for (i, rty) in sig.results.iter().enumerate() {
            let slot = b.ins().stack_load(I64, ss, (i * 8) as i32);
            vals.push(decode_slot(b, slot, *rty));
        }
    }
    Ok(())
}

/// Lower a §14 `Instantiator` `cap.call` (iface 6) to the nesting runtime ([`instantiator_rt`]) — only
/// reached when this (parent) compile has a live `Nursery` (`lower.inst.is_active()`). `op 0`
/// `instantiate(entry, off, size_log2, fuel) -> child_handle` and `op 1` `join(child_handle) ->
/// result` call the baked thunks, threading `mem_base` (the live parent window) + `trap_out`; a thunk
/// that sets the trap cell (forged handle, bad carve, child trap) propagates here like any `cap.call`.
#[allow(clippy::too_many_arguments)] // mirrors the CapCall fields + the shared lowering context
fn lower_instantiator(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    op: u32,
    sig: &FuncType,
    handle: u32,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    // §3c: the verifier checks a `cap.call`'s args against its *declared* `sig` only — it knows
    // nothing about host interfaces, so a verifier-valid module can call iface 6 with any
    // `(op, sig)` shape. The interpreter discovers a mismatch at **runtime** (handle resolution
    // first → `CapFault` on a forged handle; then per-op arg indexing → `Malformed`), but this
    // lowering dispatches on `op` **statically** and can neither index missing args nor pass
    // mistyped values through the fixed thunk ABIs. So: a call whose declared shape doesn't match
    // the op's contract (arg prefix / result types below), or whose op is unknown (the
    // interpreter's default arm is `CapFault`), lowers to an unconditional runtime `CapFault` —
    // never a compile-time rejection of a verified module. (Found by the libFuzzer `diff` target:
    // ops 5/6/7 with short args made the compile fail `Malformed`, crashing the differential —
    // nightlies Jun 19 / Jul 2 / Jul 4, ISSUES.md I16.) Extra trailing args beyond the contract
    // prefix are tolerated exactly like the interpreter (it never indexes past the op's arity).
    use ValType::{I32 as VI32, I64 as VI64};
    let contract: Option<(&[ValType], &[ValType])> = match op {
        // instantiate / spawn_coroutine / spawn_demand_coroutine: (entry, off, size_log2, fuel)
        0 | 2 | 4 => Some((&[VI64, VI64, VI64, VI64], &[VI32])),
        // *_module variants: a leading `Module` handle (i64 slot), then the same four
        5..=7 => Some((&[VI64, VI64, VI64, VI64, VI64], &[VI32])),
        // join(child) -> result
        1 => Some((&[VI32], &[VI64])),
        // coro_resume(child, value) -> (status, value)
        3 => Some((&[VI32, VI64], &[VI32, VI64])),
        // S3 lifecycle: poll / detach / kill (child) -> i32 status
        9 | 10 | 12 => Some((&[VI32], &[VI32])),
        // S2 instantiate_granted: (grant_handle, entry, off, size_log2, quota) -> child handle — the
        // grant handle rides an i64 slot (the guest widens its i32 handle), like every other arg.
        8 => Some((&[VI64, VI64, VI64, VI64, VI64], &[VI32])),
        // S2 instantiate_named: (grants_ptr, grants_n, entry, off, size_log2, quota) -> child handle
        11 => Some((&[VI64, VI64, VI64, VI64, VI64, VI64], &[VI32])),
        // STAGE1 instantiate_module_named: (module, grants_ptr, grants_n, entry, off, size_log2,
        // quota) -> child handle — op 5's leading `Module` handle then op 11's grant list + carve args.
        13 => Some((&[VI64, VI64, VI64, VI64, VI64, VI64, VI64], &[VI32])),
        _ => None,
    };
    // Width-tolerant shape check (matches the interpreter, which reads every arg as an i64 slot and
    // coerces each result to the *declared* sig type): a call is admitted when its arg count covers
    // the op's contract prefix and every prefix arg + every result is a scalar int — i32 **or** i64.
    // The exact-width `== *need` / `== res` was too strict: a chibicc guest widens all scalars to i64
    // (`int __spawn(...)` → `… -> (i64)`), so op 13's i32-result contract (and op 1/9/10/12's i32
    // child arg) never matched, and every compiled-C driver of the Instantiator fell to a CapFault
    // that the interpreter never raised. The per-arg coercions below (`slot_i64`/`slot_i32`/
    // `result_as`) reconcile the declared widths with each thunk's fixed ABI, so relaxing the gate
    // introduces no ABI mismatch. A non-scalar (or too-few args, or an unknown op) still lowers to an
    // unconditional runtime CapFault — never a compile-time rejection of a verified module.
    let is_scalar_int = |t: &ValType| matches!(t, ValType::I32 | ValType::I64);
    let shape_ok = contract.is_some_and(|(need, res)| {
        sig.params.len() >= need.len()
            && sig.params[..need.len()].iter().all(is_scalar_int)
            && sig.results.len() == res.len()
            && sig.results.iter().all(is_scalar_int)
    });
    if !shape_ok {
        emit_trap_set(b, lower, TrapKind::CapFault);
        emit_trap_propagate(b, lower);
        // Dead at runtime (the cell is already set), but keep the verifier's value accounting for
        // the rest of the block: push zeros of the *declared* result types.
        for t in &sig.results {
            let z = zero_of(b, clif_ty(*t));
            vals.push(z);
        }
        return Ok(());
    }
    let nursery = b.ins().iconst(I64, lower.inst.nursery_addr);
    let mem_base = b.use_var(lower.mem_var);
    let trap_out = b.use_var(lower.trap_var);
    match op {
        0 | 5 => {
            // instantiate(nursery, mem_base, handle:i32, module:i64, entry:i64, off:i64,
            //             size_log2:i64, fuel:i64, trap_out:i64) -> child_handle:i32. op 0 is a
            // **self** child (module = -1); op 5 (`instantiate_module`, §14 separate-module child)
            // passes a host-granted `Module` handle as its first arg and shifts the rest by one.
            let h0 = get(vals, handle)?; // the Instantiator handle (resolved for authority)
            let h = slot_i32(b, h0);
            let (modh, a0) = if op == 5 {
                (
                    slot_i64(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?),
                    1,
                )
            } else {
                (b.ins().iconst(I64, -1), 0)
            };
            let entry = slot_i64(b, get(vals, *args.get(a0).ok_or(JitError::Malformed)?)?);
            let off = slot_i64(b, get(vals, *args.get(a0 + 1).ok_or(JitError::Malformed)?)?);
            let size_log2 = slot_i64(b, get(vals, *args.get(a0 + 2).ok_or(JitError::Malformed)?)?);
            let fuel = slot_i64(b, get(vals, *args.get(a0 + 3).ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I32, I64, I64, I64, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.instantiate_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[
                    nursery, mem_base, h, modh, entry, off, size_log2, fuel, trap_out,
                ],
            );
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        1 => {
            // join(nursery, child_handle:i32, trap_out:i64) -> result:i64. The cap.call's handle
            // operand (the Instantiator) is unused here — the child handle is the first arg, and the
            // nursery owns the child table for this run.
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.join_thunk);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[nursery, child, trap_out]);
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        2 | 4 | 6 | 7 => {
            // coro_spawn(nursery, mem_base, handle:i32, module:i64, entry:i64, off:i64,
            //            size_log2:i64, fuel:i64, demand:i32, trap_out:i64) -> child_handle:i32 —
            // §14 co-fiber spawn. ops 2/4 are **self** children (module = -1); ops 6/7
            // (`spawn[_demand]_coroutine_module`) pass a `Module` handle first and shift the rest.
            // ops 4/7 demand-page the child's window for fault-driven yield.
            let h = slot_i32(b, get(vals, handle)?);
            let (modh, a0) = if op >= 6 {
                (
                    slot_i64(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?),
                    1,
                )
            } else {
                (b.ins().iconst(I64, -1), 0)
            };
            let entry = slot_i64(b, get(vals, *args.get(a0).ok_or(JitError::Malformed)?)?);
            let off = slot_i64(b, get(vals, *args.get(a0 + 1).ok_or(JitError::Malformed)?)?);
            let size_log2 = slot_i64(b, get(vals, *args.get(a0 + 2).ok_or(JitError::Malformed)?)?);
            let fuel = slot_i64(b, get(vals, *args.get(a0 + 3).ok_or(JitError::Malformed)?)?);
            let demand = b.ins().iconst(I32, if op == 4 || op == 7 { 1 } else { 0 });
            let mut tsig = module.make_signature();
            for t in [I64, I64, I32, I64, I64, I64, I64, I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.coro_spawn_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[
                    nursery, mem_base, h, modh, entry, off, size_log2, fuel, demand, trap_out,
                ],
            );
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        3 => {
            // coro_resume(nursery, mem_base, handle:i32, child:i32, value:i64, status_out:*i64,
            //             trap_out:i64) -> value:i64. Results are appended `(status:i32, value:i64)`
            // to match the op's two-result shape (like `cont.resume`).
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let h = slot_i32(b, get(vals, handle)?);
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let value = slot_i64(b, get(vals, *args.get(1).ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I32, I32, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I64));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.coro_resume_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[nursery, mem_base, h, child, value, status_ptr, trap_out],
            );
            emit_trap_propagate(b, lower);
            let value_out = b.inst_results(call)[0];
            let status64 = b.ins().stack_load(I64, ss, 0);
            let status32 = b.ins().ireduce(I32, status64);
            let status = result_as(b, status32, sig.results[0]);
            let value_out = result_as(b, value_out, sig.results[1]);
            vals.push(status);
            vals.push(value_out);
        }
        9 | 10 | 12 => {
            // S3 lifecycle: poll / detach / kill (nursery, child:i32, trap_out:i64) -> status:i32.
            // The cap.call's handle operand (the Instantiator) is unused — the child handle is arg 0,
            // and the nursery owns the child table (as for `join`).
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk_addr = match op {
                9 => lower.inst.poll_thunk,
                10 => lower.inst.detach_thunk,
                _ => lower.inst.kill_thunk,
            };
            let thunk = b.ins().iconst(I64, thunk_addr);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[nursery, child, trap_out]);
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        8 => {
            // S2 instantiate_granted(nursery, mem_base, handle:i32, grant_handle:i32, entry:i64,
            //   off:i64, size_log2:i64, fuel:i64, trap_out:i64) -> child_handle:i32. Like `instantiate`
            // (op 0) but re-grants a coordinate-free cap (arg 0) into the child's powerbox; the child
            // is a same-module child so there is no `Module` handle.
            let h = slot_i32(b, get(vals, handle)?); // the Instantiator handle (resolved for authority)
                                                     // The grant handle rides an i64 slot in the guest sig; the thunk takes it as i32.
            let grant = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let entry = slot_i64(b, get(vals, *args.get(1).ok_or(JitError::Malformed)?)?);
            let off = slot_i64(b, get(vals, *args.get(2).ok_or(JitError::Malformed)?)?);
            let size_log2 = slot_i64(b, get(vals, *args.get(3).ok_or(JitError::Malformed)?)?);
            let fuel = slot_i64(b, get(vals, *args.get(4).ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I32, I32, I64, I64, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.instantiate_granted_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[
                    nursery, mem_base, h, grant, entry, off, size_log2, fuel, trap_out,
                ],
            );
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        11 => {
            // S2 instantiate_named(nursery, mem_base, mem_size:i64, handle:i32, grants_ptr:i64,
            //   grants_n:i64, entry:i64, off:i64, size_log2:i64, fuel:i64, trap_out:i64) -> handle:i32.
            // Like op 8 but the child's caps come from a grant-record list in the window (`mem_size`
            // bounds the host-side reads); no positional grant arg, same-module child.
            let h = slot_i32(b, get(vals, handle)?);
            let mem_size = b.ins().iconst(I64, lower.mapped as i64);
            let grants_ptr = slot_i64(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let grants_n = slot_i64(b, get(vals, *args.get(1).ok_or(JitError::Malformed)?)?);
            let entry = slot_i64(b, get(vals, *args.get(2).ok_or(JitError::Malformed)?)?);
            let off = slot_i64(b, get(vals, *args.get(3).ok_or(JitError::Malformed)?)?);
            let size_log2 = slot_i64(b, get(vals, *args.get(4).ok_or(JitError::Malformed)?)?);
            let fuel = slot_i64(b, get(vals, *args.get(5).ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I32, I64, I64, I64, I64, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.inst.instantiate_named_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[
                    nursery, mem_base, mem_size, h, grants_ptr, grants_n, entry, off, size_log2,
                    fuel, trap_out,
                ],
            );
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        13 => {
            // STAGE1 instantiate_module_named(nursery, mem_base, mem_size, handle:i32, module:i64,
            //   grants_ptr:i64, grants_n:i64, entry:i64, off:i64, size_log2:i64, fuel:i64,
            //   trap_out:i64) -> handle:i32. Op 5's leading `Module` handle then op 11's grant-list
            // args — runs a foreign module with a by-name granted powerbox (the shell "exec").
            let h = slot_i32(b, get(vals, handle)?);
            let mem_size = b.ins().iconst(I64, lower.mapped as i64);
            let modh = slot_i64(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let grants_ptr = slot_i64(b, get(vals, *args.get(1).ok_or(JitError::Malformed)?)?);
            let grants_n = slot_i64(b, get(vals, *args.get(2).ok_or(JitError::Malformed)?)?);
            let entry = slot_i64(b, get(vals, *args.get(3).ok_or(JitError::Malformed)?)?);
            let off = slot_i64(b, get(vals, *args.get(4).ok_or(JitError::Malformed)?)?);
            let size_log2 = slot_i64(b, get(vals, *args.get(5).ok_or(JitError::Malformed)?)?);
            let fuel = slot_i64(b, get(vals, *args.get(6).ok_or(JitError::Malformed)?)?);
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I32, I64, I64, I64, I64, I64, I64, I64, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b
                .ins()
                .iconst(I64, lower.inst.instantiate_module_named_thunk);
            let call = b.ins().call_indirect(
                tref,
                thunk,
                &[
                    nursery, mem_base, mem_size, h, modh, grants_ptr, grants_n, entry, off,
                    size_log2, fuel, trap_out,
                ],
            );
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        // Unknown ops were rejected by the shape check above (→ runtime CapFault, matching the
        // interpreter's default arm) — this match only sees contract-validated ops.
        _ => unreachable!("shape check admitted an unknown Instantiator op"),
    }
    Ok(())
}

fn get(vals: &[Value], i: u32) -> Result<Value, JitError> {
    vals.get(i as usize).copied().ok_or(JitError::Malformed)
}

/// Widen a fetched cap.call arg to the `i64` slot an Instantiator thunk param takes — the JIT analogue
/// of the interpreter reading every arg as `Reg::i64()` off a **sign-extended** slot (`from_i32`). A
/// chibicc-emitted call already passes i64 (passthrough); a frontend that declares a slot arg `i32`
/// gets sign-extended, matching the interp exactly. Keeps the two backends in lockstep even though the
/// verifier only checks a cap.call against its *declared* sig (§3c), not the host op's contract.
fn slot_i64(b: &mut FunctionBuilder, v: Value) -> Value {
    if b.func.dfg.value_type(v) == I32 {
        b.ins().sextend(I64, v)
    } else {
        v
    }
}

/// Narrow a fetched cap.call arg to the `i32` a handle/child thunk param takes. chibicc widens every
/// scalar to i64, but handles cross as i32 (the host table masks them); an already-i32 value passes
/// through. Mirrors the interpreter's `get_i32` on the same slot.
fn slot_i32(b: &mut FunctionBuilder, v: Value) -> Value {
    if b.func.dfg.value_type(v) == I64 {
        b.ins().ireduce(I32, v)
    } else {
        v
    }
}

/// Coerce a thunk result to the guest's *declared* result type — the JIT analogue of the interpreter's
/// `slot_to_val(ty, slot)`. The thunks return canonical widths (an i32 child handle / status, an i64
/// join value); a chibicc sig declares i64 for an `int`-returning call, so **sign-extend** the i32 up
/// (a spawn yields a small non-negative handle or a negative errno — sign-extension matches both the
/// interp's i64 slot and C's `(int)`→`long`). Narrows the reverse case for completeness.
fn result_as(b: &mut FunctionBuilder, v: Value, declared: ValType) -> Value {
    let want = clif_ty(declared);
    let have = b.func.dfg.value_type(v);
    if have == want {
        v
    } else if want == I64 && have == I32 {
        b.ins().sextend(I64, v)
    } else if want == I32 && have == I64 {
        b.ins().ireduce(I32, v)
    } else {
        v
    }
}

/// The §4 confinement lowering (invariant I1, **branchless confinement**, D63): bounds-test the
/// final effective address `addr + offset` against `reserved − offset − width`, then
/// `select_spectre_guard(oob, guard_offset, sub_base + addr + offset)` and add the base, giving the
/// physical address `mem_base + (oob ? guard_offset : sub_base + addr + offset)`. An out-of-bounds
/// access is redirected to `guard_offset` — the offset of the enclosing window's trailing guard page
/// (`round_up(win_reserved, page)`) — so the access *itself* faults there (`MemoryFault`, the
/// architectural fault, mirroring `svm_mask::Window::checked`), with no per-access branch. The
/// `select_spectre_guard` is also a speculation barrier: a misspeculated OOB access likewise receives
/// `guard_offset`, confining it to the guard — the Spectre-v1 hardening (§4, D63; supersedes the D38
/// `trapnz` + `& mask` clamp). This is uniform for top-level and §14 nested (`sub_base != 0`) windows:
/// a nested child redirects to the **parent's** guard (its own slice has committed parent memory on
/// both sides), so an out-of-child access lands on the parent guard, never in the parent.
///
/// When `elide` is set the check and clamp are both dropped — but **only** the caller's
/// [`in_window`] proof (the address is provably `< size`) may set it, so the unclamped
/// `addr + offset` already stays in `[0, size)` (a proven bound holds speculatively too:
/// the proof is over data dependencies, e.g. `(i & K)*W`). This is the
/// elide-when-provably-bounded half of §1a (D36–D38); a wrong proof is a confinement
/// escape, caught by the escape-oracle (final-memory differential, §18). The `+ sub_base`
/// is independent of elision (it shifts the whole `[0, size)` child window into its parent
/// slice) and is itself elided when `sub_base == 0`.
///
/// Offset from `mem_base` of the enclosing window's trailing guard page (`round_up(win_reserved,
/// page)`), the always-`PROT_NONE` redirect target for the D63 branchless out-of-bounds fault. `0`
/// (unused) when the window has no reservation.
fn guard_offset_of(win_reserved: u64) -> u64 {
    if win_reserved == 0 {
        return 0;
    }
    let page = mem::page_size() as u64;
    (win_reserved + page - 1) & !(page - 1)
}

fn mask_addr(
    b: &mut FunctionBuilder,
    lower: &Lower,
    addr: Value,
    offset: u64,
    elide: bool,
    width: u32,
) -> Value {
    // Fold the immediate only when non-zero, so an offset-0 access keeps a minimal address
    // expression (helps Cranelift's GVN / store-to-load forwarding recognize equal addresses).
    let eff = if offset == 0 {
        addr
    } else {
        let off = b.ins().iconst(I64, offset as i64);
        b.ins().iadd(addr, off)
    };
    // A non-elided access is confined by a bounds test + a branchless spectre-guard redirect (invariant
    // I1, **branchless confinement**, §4/D63): the access `[addr+offset, addr+offset+width)` must lie in
    // the reserved domain `[0, reserved)`, else a clear `MemoryFault` fires *at the offending access*
    // (vs the old mask silently wrapping a wild access to some other in-window byte). The bounds test
    // compares the dynamic `addr` against the compile-time constant `reserved - offset - width`, so it
    // never itself overflows; an out-of-bounds address is redirected (via `select_spectre_guard`, no
    // branch) to the enclosing window's trailing guard page and faults there. Within the reservation,
    // the committed-ness of `[0, mapped)` (and any page the guest `grow`s into the tail) is enforced by
    // the `PROT_NONE` guard region — a stray `[mapped, reserved)` access faults there. This mirrors the
    // interpreter's `confine_checked` (reserved-domain bound) + `check_prot` (per-page) — the redirect
    // is invisible to that mirror (it only fires on an access the interpreter also faults; the
    // interpreter doesn't speculate) — so the two agree on the same fault under the §18 escape oracle.
    // A §14 nested child (`sub_base != 0`) bounds-tests the same way, shifts the in-bounds offset into
    // its parent slice (`+ sub_base`), and redirects an out-of-child access to the parent guard, so it
    // too faults out-of-child instead of aliasing parent memory. Elided accesses (proven `< reserved`)
    // need no check. The no-memory / degenerate `mask == 0` case keeps the mask (a 1-byte / memoryless
    // window has no reservation to bound against).
    if !elide && lower.mask != 0 {
        // `reserved = mask + 1` (a power of two ≤ 2^63); the reservation this window is confined to.
        let reserved = lower.mask.wrapping_add(1);
        let need = (width as u64).saturating_add(offset);
        // Out-of-bounds test: `addr > reserved − offset − width` ⇒ some byte of the access falls
        // outside `[0, reserved)`. The compare is against a compile-time constant, so it never itself
        // overflows.
        let oob = match reserved.checked_sub(need) {
            Some(limit) => {
                let lim = b.ins().iconst(I64, limit as i64);
                b.ins().icmp(IntCC::UnsignedGreaterThan, addr, lim)
            }
            // `offset + width` alone exceeds the reservation ⇒ no `addr` is in-bounds; always fault.
            None => b.ins().iconst(I8, 1),
        };
        // **Branchless confinement** (invariant I1, D63), matching Wasmtime's spectre-guarded heap and
        // uniform for top-level *and* §14 nested windows. No `trapnz`: shift the (proven in-bounds when
        // `!oob`) offset into its window slice (`+ sub_base`, elided top-level), then
        // `select_spectre_guard(oob, guard_offset, …)` redirects an out-of-bounds access to
        // `guard_offset` — the offset of the **enclosing window's trailing guard page** from `mem_base`
        // — so the load/store *itself* faults there (SIGSEGV → `MemoryFault`, unwound by the guard
        // handler like any in-window guard fault), with no per-access branch.
        //
        // `guard_offset = round_up(win_reserved, page)` is always `PROT_NONE` (`GuestWindow` maps
        // `[round_up(mapped,page), round_up(reserved,page)+page)` inaccessible). For a top-level window
        // `win_reserved == reserved`, so this is the window's own guard; for a nested sub-window it is
        // the **parent's** guard — the only guaranteed fault site, since the child slice has committed
        // parent memory on both sides (redirecting to the child's own `reserved` could alias valid
        // parent memory: a child→parent escape). Redirecting the *whole* physical offset (past
        // `+ sub_base`) means an out-of-child access lands on the parent guard, never in the parent.
        //
        // `select_spectre_guard` is a **speculation barrier** (a data-dependent cmov, not a predicted
        // branch): a *misspeculated* OOB access also receives `guard_offset`, so it can never
        // speculatively read past the window — the same Spectre-v1 confinement the old AND mask gave,
        // via the identical primitive Wasmtime uses (in-process nesting is defense-in-depth, not a
        // promised Spectre boundary — DESIGN.md §2a). Dropping the per-access branch **and** the
        // memory-operand AND (a 40-bit mask is not an x86 immediate, so it was a RIP-relative load each
        // access) closes the residual array-kernel gap to Wasmtime-w64 (matmul 1.06→0.94,
        // matmul_eb 1.34→0.97 svm÷wt64; see the `confine` harness / TRAP_CONFINEMENT.md). The
        // clear-`MemoryFault`-at-the-offending-access property of §4/D38 is preserved — only the
        // mechanism (guard-page fault vs `ud2`) differs.
        let shifted = if lower.sub_base == 0 {
            eff
        } else {
            let sb = b.ins().iconst(I64, lower.sub_base as i64);
            b.ins().iadd(eff, sb)
        };
        let guard = b.ins().iconst(I64, lower.guard_offset as i64);
        let confined = b.ins().select_spectre_guard(oob, guard, shifted);
        let base = b.use_var(lower.mem_var);
        return b.ins().iadd(base, confined);
    }
    // Elided (proven in-window) or the degenerate `mask == 0` (memoryless) path: no bounds check, and
    // the mask kept only for `mask == 0`.
    let confined = if elide {
        eff
    } else {
        let m = b.ins().iconst(I64, lower.mask as i64);
        b.ins().band(eff, m)
    };
    // §14 sub-window: shift the confined child offset into its parent slice. Elided (no add) for a
    // top-level window so ordinary codegen is byte-identical to before nesting existed.
    let confined = if lower.sub_base == 0 {
        confined
    } else {
        let sb = b.ins().iconst(I64, lower.sub_base as i64);
        b.ins().iadd(confined, sb)
    };
    let base = b.use_var(lower.mem_var);
    b.ins().iadd(base, confined)
}

/// **Bulk-span backed-extent probe (ISSUES.md I21).** Fault a bulk span that overruns the
/// currently-*backed* region **before** the libcall runs. [`confine_span`] only bounds the span
/// against `reserved`, delegating the `[mapped, reserved)` guard hole to the libcall's own accesses
/// — but that leaks in two ways the interpreter (`Mem::check_prot_span`, which validates every page
/// up front) does not: (1) libc `memcpy`/`memmove` **short-circuit `dst == src`**, so a self-copy
/// overrunning `mapped` never faults; (2) the libcall writes a prefix before hitting the guard, so a
/// faulting run diverges from the interpreter's fault-before-any-write. Both are interp↔JIT
/// divergences (§3 parity), not escapes (every byte stays in `[0, reserved)`).
///
/// The fix is a guarded 1-byte read of the span's **last** byte (`phys + len − 1`, where `phys` is
/// [`confine_span`]'s confined base). It faults [`TrapKind::MemoryFault`] on the `PROT_NONE` guard
/// exactly where the interpreter faults — **consulting the live page tables**, so it honors guest
/// `grow` (unlike the compile-time `Lower::mapped`, which would over-fault a grown window). For the
/// contiguous backed prefix (the production model + the spec window), last-byte-in-bounds ⟺
/// whole-span-in-bounds, matching the interpreter's own `last < mapped` fast path. Emitted per span
/// (`dst` and `src`) before the copy, so no partial write survives. `len == 0` skips the probe
/// entirely (a branch on `len != 0`), keeping a zero-length op inert even at a wild pointer (D62).
///
/// Residual (documented, not silently dropped): a bulk op whose span straddles a guest-created
/// **interior** hole (`unmap`) or a read-only page mid-span is not caught by a last-byte read probe
/// — the same boundary the interpreter's contiguous fast path assumes away. Left to a per-page probe
/// loop if a real workload needs bulk ops over a deliberately-punched window.
fn probe_span(b: &mut FunctionBuilder, phys: Value, len: Value) {
    let do_probe = b.create_block();
    let after = b.create_block();
    let nonzero = b.ins().icmp_imm(IntCC::NotEqual, len, 0);
    b.ins().brif(nonzero, do_probe, &[], after, &[]);

    b.switch_to_block(do_probe);
    b.seal_block(do_probe);
    let one = b.ins().iconst(I64, 1);
    let last_off = b.ins().isub(len, one);
    let last = b.ins().iadd(phys, last_off);
    // A may-trap load: it faults `MemoryFault` on the guard page if the span overruns the backed
    // region. The result is unused, but the load is preserved because it can trap (side-effecting).
    b.ins().load(I8, mem_flags(), last, 0);
    b.ins().jump(after, &[]);

    b.switch_to_block(after);
    b.seal_block(after);
}

/// Confine a **dynamic-length span** `[ptr, ptr+len)` to the reserved domain `[0, reserved)` and
/// return its physical base `mem_base + (ptr & mask) + sub_base` — the bulk-memory hinge (D62). Where
/// [`mask_addr`] confines one fixed-width access, this confines a whole span with a *single* range
/// check: it traps [`TrapKind::MemoryFault`] (native `trapnz`) when `len != 0 && (len > reserved ||
/// ptr > reserved − len)`, i.e. any byte of the span would fall outside `[0, reserved)`. The two
/// sub-checks avoid overflow: `len > reserved` catches an oversized (or negative-as-u64) length before
/// the `reserved − len` subtraction can wrap. `len == 0` is a no-op span that never faults (matching
/// the interpreter and C `memcpy(_,_,0)`), so a 0-length op on a wild pointer is inert.
///
/// The returned base is then Spectre-clamped (`& mask`) exactly as in [`mask_addr`]: architecturally a
/// no-op (the check proved `ptr < reserved`), but on a *mispredicted* path it pins the copy's base
/// inside `[0, reserved)`. As with `mask_addr`, this matches Wasmtime's bounds-checked `memory.copy`
/// posture ("as secure as wasm", DESIGN.md §1a): the length is checked, the base is confined, and the
/// bulk copy itself is a libcall the CPU does not speculate byte-by-byte — so no per-byte clamp is
/// needed. Callers pass the **same** `len` to the copy, so both spans (dst and src) are checked.
fn confine_span(b: &mut FunctionBuilder, lower: &Lower, ptr: Value, len: Value) -> Value {
    let reserved = lower.mask.wrapping_add(1);
    let resv = b.ins().iconst(I64, reserved as i64);
    let oob_len = b.ins().icmp(IntCC::UnsignedGreaterThan, len, resv);
    let rem = b.ins().isub(resv, len); // reserved − len; meaningful only when !oob_len
    let oob_ptr = b.ins().icmp(IntCC::UnsignedGreaterThan, ptr, rem);
    let oob_span = b.ins().bor(oob_len, oob_ptr);
    let zero = b.ins().iconst(I64, 0);
    let nonzero = b.ins().icmp(IntCC::NotEqual, len, zero);
    let oob = b.ins().band(nonzero, oob_span);
    b.ins()
        .trapnz(oob, cranelift_codegen::ir::TrapCode::HEAP_OUT_OF_BOUNDS);
    let m = b.ins().iconst(I64, lower.mask as i64);
    let clamped = b.ins().band(ptr, m);
    let shifted = if lower.sub_base == 0 {
        clamped
    } else {
        let sb = b.ins().iconst(I64, lower.sub_base as i64);
        b.ins().iadd(clamped, sb)
    };
    let base = b.use_var(lower.mem_var);
    b.ins().iadd(base, shifted)
}

/// Unknown upper bound — the value may be anything (so its accesses must be masked).
const UB_TOP: u64 = u64::MAX;

/// The recorded upper bound of IR value `i` (unknown if out of range — defensive).
fn ub_at(ubs: &[u64], i: u32) -> u64 {
    ubs.get(i as usize).copied().unwrap_or(UB_TOP)
}

/// A **sound, conservative upper bound** on an SSA value's unsigned (`u64`) magnitude, used
/// only to decide mask elision. Every rule must never under-estimate the real maximum;
/// anything not modelled returns [`UB_TOP`]. Lower bounds are irrelevant (a `u64` is `≥ 0`),
/// so only the upper bound is tracked. Indexed like the value map (block params = `UB_TOP`).
fn ub_of(inst: &Inst, ubs: &[u64]) -> u64 {
    let ub = |i: u32| ubs.get(i as usize).copied().unwrap_or(UB_TOP);
    match inst {
        Inst::ConstI64(c) => *c as u64,
        Inst::ConstI32(c) => *c as u32 as u64,
        Inst::IntBin { op, a, b, .. } => {
            let (x, y) = (ub(*a), ub(*b));
            match op {
                // a & b ≤ min(a, b); a|b, a^b, a+b ≤ a + b; a*b ≤ a * b (wrap ⇒ Top).
                BinOp::And => x.min(y),
                BinOp::Add | BinOp::Or | BinOp::Xor => x.checked_add(y).unwrap_or(UB_TOP),
                BinOp::Mul => x.checked_mul(y).unwrap_or(UB_TOP),
                _ => UB_TOP,
            }
        }
        // Zero-extend: the i64 value is the (≤ u32::MAX) source, no wider.
        Inst::Convert {
            op: ConvOp::ExtendI32U,
            a,
        } => ub(*a).min(0xFFFF_FFFF),
        Inst::Convert {
            op: ConvOp::WrapI64,
            ..
        } => 0xFFFF_FFFF,
        _ => UB_TOP,
    }
}

/// True iff every access `[addr+offset, addr+offset+width)` is provably within `[0, size)`
/// given `addr ≤ addr_ub` — i.e. the mask is redundant and can be elided. Saturating/checked
/// throughout so an overflow can only make this *false* (fall back to masking), never escape.
fn in_window(addr_ub: u64, offset: u64, width: u32, size: u64) -> bool {
    match addr_ub
        .checked_add(offset)
        .and_then(|s| s.checked_add(width as u64))
    {
        Some(top) => top <= size,
        None => false,
    }
}

/// Little-endian, may-trap memory access flags (the window is host memory; the guard
/// margin absorbs width overrun, so this never faults in practice).
fn mem_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf
}

/// The CLIF type holding `width` raw bytes.
fn width_ty(width: u32) -> Type {
    match width {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    }
}

fn lower_load(b: &mut FunctionBuilder, op: LoadOp, phys: Value) -> Value {
    let (_, rty, width, signed) = op.info();
    // Float loads read the float type directly (no extension).
    if matches!(rty, ValType::F32 | ValType::F64) {
        return b.ins().load(clif_ty(rty), mem_flags(), phys, 0);
    }
    let load_ty = width_ty(width);
    let raw = b.ins().load(load_ty, mem_flags(), phys, 0);
    let result_ty = clif_ty(rty);
    if load_ty == result_ty {
        raw
    } else if signed {
        b.ins().sextend(result_ty, raw) // narrow signed load: sign-extend
    } else {
        b.ins().uextend(result_ty, raw) // narrow unsigned load: zero-extend
    }
}

fn lower_store(b: &mut FunctionBuilder, op: StoreOp, phys: Value, value: Value) {
    let (_, vty, width) = op.info();
    // Float stores write the float bits directly.
    if matches!(vty, ValType::F32 | ValType::F64) {
        b.ins().store(mem_flags(), value, phys, 0);
        return;
    }
    let store_ty = width_ty(width);
    // Narrow stores keep only the low `width` bytes (matches the interpreter).
    let v = if b.func.dfg.value_type(value) == store_ty {
        value
    } else {
        b.ins().ireduce(store_ty, value)
    };
    b.ins().store(mem_flags(), v, phys, 0);
}

/// Access width (and natural-alignment requirement) of a §12 atomic `ty`.
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Memory flags for an atomic access: little-endian (the window is LE) and aligned — a preceding
/// [`guard_atomic_align`] traps a misaligned address, so the hardware atomic only ever sees a
/// naturally-aligned one.
fn atomic_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf.set_aligned();
    mf
}

/// Map an IR atomic RMW op to Cranelift's.
fn clif_rmw_op(op: AtomicRmwOp) -> ClifRmwOp {
    match op {
        AtomicRmwOp::Add => ClifRmwOp::Add,
        AtomicRmwOp::Sub => ClifRmwOp::Sub,
        AtomicRmwOp::And => ClifRmwOp::And,
        AtomicRmwOp::Or => ClifRmwOp::Or,
        AtomicRmwOp::Xor => ClifRmwOp::Xor,
        AtomicRmwOp::Xchg => ClifRmwOp::Xchg,
    }
}

/// Trap (`MemoryFault`) if physical address `phys` is not `width`-aligned, else fall through —
/// mirrors the §12 interpreter's `check_align`. `atomic_*` lowerings require natural alignment
/// (e.g. aarch64 `LDAXR`/`STLXR` fault otherwise), so this precedes every atomic. Leaves the
/// builder positioned in the aligned ("ok") block.
fn guard_atomic_align(b: &mut FunctionBuilder, lower: &Lower, phys: Value, width: u32) {
    if width <= 1 {
        return;
    }
    let rem = b.ins().band_imm(phys, (width - 1) as i64);
    let aligned = b.ins().icmp_imm(IntCC::Equal, rem, 0);
    let ok = b.create_block();
    let bad = b.create_block();
    b.ins().brif(aligned, ok, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::MemoryFault);
    b.switch_to_block(ok);
    b.seal_block(ok);
}

/// Decode an `i64` calling-convention slot to a value of IR type `ty`.
fn decode_slot(b: &mut FunctionBuilder, slot: Value, ty: ValType) -> Value {
    match ty {
        ValType::I64 | ValType::Ref => slot, // `ref` is an opaque i64 in the cap ABI
        ValType::I32 => b.ins().ireduce(I32, slot),
        ValType::F32 => {
            let i = b.ins().ireduce(I32, slot);
            b.ins().bitcast(F32, MemFlags::new(), i)
        }
        ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), slot),
        // `v128` cap-ABI slots are out of MVP scope (§17); a zero vector keeps this total.
        ValType::V128 => {
            let z = b.ins().iconst(I8, 0);
            b.ins().splat(I8X16, z)
        }
    }
}

/// Encode a value into its `i64` calling-convention slot (the harness reads back the
/// low 32 bits for i32/f32 results).
fn encode_slot(b: &mut FunctionBuilder, v: Value) -> Value {
    match b.func.dfg.value_type(v) {
        I64 => v,
        I32 => b.ins().uextend(I64, v),
        F32 => {
            let i = b.ins().bitcast(I32, MemFlags::new(), v);
            b.ins().uextend(I64, i)
        }
        F64 => b.ins().bitcast(I64, MemFlags::new(), v),
        _ => v,
    }
}

/// Map a vector float op to the scalar [`FBinOp`]/[`FUnOp`] so vector lanes lower exactly
/// like scalars (§17/D58).
fn vf_bin(op: VFloatBinOp) -> FBinOp {
    match op {
        VFloatBinOp::Add => FBinOp::Add,
        VFloatBinOp::Sub => FBinOp::Sub,
        VFloatBinOp::Mul => FBinOp::Mul,
        VFloatBinOp::Div => FBinOp::Div,
        VFloatBinOp::Min => FBinOp::Min,
        VFloatBinOp::Max => FBinOp::Max,
    }
}
fn vf_un(op: VFloatUnOp) -> FUnOp {
    match op {
        VFloatUnOp::Abs => FUnOp::Abs,
        VFloatUnOp::Neg => FUnOp::Neg,
        VFloatUnOp::Sqrt => FUnOp::Sqrt,
        VFloatUnOp::Ceil => FUnOp::Ceil,
        VFloatUnOp::Floor => FUnOp::Floor,
        VFloatUnOp::Trunc => FUnOp::Trunc,
        VFloatUnOp::Nearest => FUnOp::Nearest,
    }
}

fn float_bin(b: &mut FunctionBuilder, op: FBinOp, x: Value, y: Value) -> Value {
    match op {
        FBinOp::Add => b.ins().fadd(x, y),
        FBinOp::Sub => b.ins().fsub(x, y),
        FBinOp::Mul => b.ins().fmul(x, y),
        FBinOp::Div => b.ins().fdiv(x, y),
        // The IR defines `min`/`max` to yield a **canonical** NaN on any NaN input — the reference
        // interpreter's `fmin`/`fmax` force `0x7FC0_0000` / `0x7FF8_..` (§18 oracle). Cranelift's
        // `fmin`/`fmax` instead *propagate* the input NaN's payload and sign, so without a fixup the
        // JIT and interp would store different NaN bits and the byte-exact escape-oracle window
        // comparison would diverge. (Add/Sub/Mul/Div need no fixup: both backends emit the same host
        // float instruction, so their NaN results already match — only the hand-written min/max
        // canonicalizes.) Canonicalize scalar and per-lane v128 alike.
        FBinOp::Min => {
            let r = b.ins().fmin(x, y);
            canonicalize_nan(b, r)
        }
        FBinOp::Max => {
            let r = b.ins().fmax(x, y);
            canonicalize_nan(b, r)
        }
        FBinOp::Copysign => b.ins().fcopysign(x, y),
    }
}

/// Replace any NaN in `r` with the IR's canonical quiet NaN — `0x7FC0_0000` (f32) /
/// `0x7FF8_0000_0000_0000` (f64) — matching the reference interpreter. Works for a scalar float and,
/// lane-wise, for a `v128` float vector (`Unordered(r, r)` is true exactly on the NaN lanes).
fn canonicalize_nan(b: &mut FunctionBuilder, r: Value) -> Value {
    let ty = b.func.dfg.value_type(r);
    // `Unordered(r, r)` is true exactly where `r` is NaN (a per-lane mask for vectors).
    let is_nan = b.ins().fcmp(FloatCC::Unordered, r, r);
    if ty.is_vector() {
        // Blend in the integer-vector domain so the fcmp mask and the operands share a type
        // (`bitselect` requires it): fcmp on F32X4 / F64X2 yields the I32X4 / I64X2 lane mask.
        let (ity, bits): (Type, i64) = if ty.lane_type() == F32 {
            (I32X4, 0x7FC0_0000)
        } else {
            (I64X2, 0x7FF8_0000_0000_0000u64 as i64)
        };
        let canon_lane = b.ins().iconst(ity.lane_type(), bits);
        let canon = b.ins().splat(ity, canon_lane);
        let ri = vcast(b, r, ity);
        let sel = b.ins().bitselect(is_nan, canon, ri);
        vcast(b, sel, ty)
    } else {
        let canon = if ty == F32 {
            b.ins().f32const(f32::from_bits(0x7FC0_0000))
        } else {
            b.ins().f64const(f64::from_bits(0x7FF8_0000_0000_0000))
        };
        b.ins().select(is_nan, canon, r)
    }
}

fn float_un(b: &mut FunctionBuilder, op: FUnOp, x: Value) -> Value {
    match op {
        FUnOp::Abs => b.ins().fabs(x),
        FUnOp::Neg => b.ins().fneg(x),
        FUnOp::Sqrt => b.ins().sqrt(x),
        FUnOp::Ceil => b.ins().ceil(x),
        FUnOp::Floor => b.ins().floor(x),
        FUnOp::Trunc => b.ins().trunc(x),
        FUnOp::Nearest => b.ins().nearest(x),
    }
}

fn float_cc(op: FCmpOp) -> FloatCC {
    match op {
        FCmpOp::Eq => FloatCC::Equal,
        FCmpOp::Ne => FloatCC::NotEqual, // unordered ≠ (NaN ne x is true), wasm semantics
        FCmpOp::Lt => FloatCC::LessThan,
        FCmpOp::Le => FloatCC::LessThanOrEqual,
        FCmpOp::Gt => FloatCC::GreaterThan,
        FCmpOp::Ge => FloatCC::GreaterThanOrEqual,
    }
}

/// Map IR edge args to CLIF block-call args (`BlockArg`, the 0.132 block-call type).
fn map_args(vals: &[Value], args: &[u32]) -> Result<Vec<BlockArg>, JitError> {
    args.iter()
        .map(|a| get(vals, *a).map(BlockArg::from))
        .collect()
}

fn int_bin(b: &mut FunctionBuilder, op: BinOp, x: Value, y: Value) -> Value {
    match op {
        BinOp::Add => b.ins().iadd(x, y),
        BinOp::Sub => b.ins().isub(x, y),
        BinOp::Mul => b.ins().imul(x, y),
        BinOp::And => b.ins().band(x, y),
        BinOp::Or => b.ins().bor(x, y),
        BinOp::Xor => b.ins().bxor(x, y),
        BinOp::Shl => b.ins().ishl(x, y),
        BinOp::ShrS => b.ins().sshr(x, y),
        BinOp::ShrU => b.ins().ushr(x, y),
        BinOp::Rotl => b.ins().rotl(x, y),
        BinOp::Rotr => b.ins().rotr(x, y),
        // div/rem are guarded and lowered by `lower_div_rem`, never here.
        BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => unreachable!("guarded elsewhere"),
    }
}

/// Lower a trapping `div`/`rem` with explicit guards: a zero divisor traps
/// `DivByZero`, and signed `div` of `INT_MIN / -1` traps `IntOverflow` (matching the
/// interpreter). On the non-trapping path the division runs on a value Cranelift's
/// `sdiv`/`srem` will not fault on (so the hardware op never traps). Returns the
/// quotient/remainder in the final ("safe") block.
fn lower_div_rem(
    b: &mut FunctionBuilder,
    lower: &Lower,
    ty: IntTy,
    op: BinOp,
    x: Value,
    y: Value,
) -> Value {
    let ity = int_clif_ty(ty);
    // Trap on a zero divisor.
    let is_zero = b.ins().icmp_imm(IntCC::Equal, y, 0);
    let after_zero = b.create_block();
    let dz = b.create_block();
    b.ins().brif(is_zero, dz, &[], after_zero, &[]);
    b.switch_to_block(dz);
    b.seal_block(dz);
    emit_trap(b, lower, TrapKind::DivByZero);
    b.switch_to_block(after_zero);
    b.seal_block(after_zero);

    // Signed div additionally traps on INT_MIN / -1 (overflow).
    if op == BinOp::DivS {
        let min = match ty {
            IntTy::I32 => i32::MIN as i64,
            IntTy::I64 => i64::MIN,
        };
        let x_is_min = b.ins().icmp_imm(IntCC::Equal, x, min);
        let y_is_neg1 = b.ins().icmp_imm(IntCC::Equal, y, -1);
        let overflow = b.ins().band(x_is_min, y_is_neg1);
        let after_ov = b.create_block();
        let ov = b.create_block();
        b.ins().brif(overflow, ov, &[], after_ov, &[]);
        b.switch_to_block(ov);
        b.seal_block(ov);
        emit_trap(b, lower, TrapKind::IntOverflow);
        b.switch_to_block(after_ov);
        b.seal_block(after_ov);
    }

    let _ = ity;
    match op {
        BinOp::DivS => b.ins().sdiv(x, y),
        BinOp::DivU => b.ins().udiv(x, y),
        // `INT_MIN % -1 == 0` is representable so `rem_s` does not trap (only `div_s`
        // does); Cranelift's `srem` yields 0 here, matching the interpreter.
        BinOp::RemS => b.ins().srem(x, y),
        BinOp::RemU => b.ins().urem(x, y),
        _ => unreachable!("non-div/rem routed here"),
    }
}

/// Lower a trapping float→int (`trunc`): trap `BadConversion` on NaN or out-of-range,
/// else convert. The bounds are the interpreter's exact bounds (computed in `f64`, so
/// `f32` is promoted first), so the JIT and interpreter trap on identical inputs.
fn lower_trunc_trap(
    b: &mut FunctionBuilder,
    lower: &Lower,
    x: Value,
    from: FloatTy,
    to: IntTy,
    signed: bool,
) -> Value {
    // Promote f32 -> f64 (exact) so one set of bounds covers both.
    let xf = match from {
        FloatTy::F32 => b.ins().fpromote(F64, x),
        FloatTy::F64 => x,
    };
    // (lower bound, upper bound, lower-inclusive). Ordered comparisons are false for
    // NaN, so NaN falls out of range and traps — no separate NaN check needed.
    let (lo, hi, lo_incl) = match (to, signed) {
        (IntTy::I32, true) => (-2_147_483_649.0_f64, 2_147_483_648.0_f64, false),
        (IntTy::I32, false) => (-1.0_f64, 4_294_967_296.0_f64, false),
        (IntTy::I64, true) => (
            -9_223_372_036_854_775_808.0_f64,
            9_223_372_036_854_775_808.0_f64,
            true,
        ),
        (IntTy::I64, false) => (-1.0_f64, 18_446_744_073_709_551_616.0_f64, false),
    };
    let lo_c = b.ins().f64const(lo);
    let hi_c = b.ins().f64const(hi);
    let ge_lo = if lo_incl {
        b.ins().fcmp(FloatCC::GreaterThanOrEqual, xf, lo_c)
    } else {
        b.ins().fcmp(FloatCC::GreaterThan, xf, lo_c)
    };
    let lt_hi = b.ins().fcmp(FloatCC::LessThan, xf, hi_c);
    let in_range = b.ins().band(ge_lo, lt_hi);

    let ok = b.create_block();
    let bad = b.create_block();
    b.ins().brif(in_range, ok, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::BadConversion);
    b.switch_to_block(ok);
    b.seal_block(ok);
    // In range: the saturating cast is exact and never faults.
    let ity = int_clif_ty(to);
    if signed {
        b.ins().fcvt_to_sint_sat(ity, x)
    } else {
        b.ins().fcvt_to_uint_sat(ity, x)
    }
}

fn int_cc(op: svm_ir::CmpOp) -> IntCC {
    use svm_ir::CmpOp::*;
    match op {
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        LtS => IntCC::SignedLessThan,
        LtU => IntCC::UnsignedLessThan,
        LeS => IntCC::SignedLessThanOrEqual,
        LeU => IntCC::UnsignedLessThanOrEqual,
        GtS => IntCC::SignedGreaterThan,
        GtU => IntCC::UnsignedGreaterThan,
        GeS => IntCC::SignedGreaterThanOrEqual,
        GeU => IntCC::UnsignedGreaterThanOrEqual,
    }
}
