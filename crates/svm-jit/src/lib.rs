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
//! ## The masking lowering (§4, invariant I1)
//! Every access masks the **final effective address** into the window —
//! `(addr + offset) & (size - 1)` — then adds the window base. This is exactly
//! [`svm_mask::Window::confine`] (the isolated, separately-fuzzed spec), so the JIT
//! and that unit lower the same arithmetic. The window allocation carries a small
//! guard margin so a masked base near the top plus the access width never escapes the
//! allocation (a real deployment uses guard *pages* + a fault for the width overrun).
//!
//! **Mask elision (§1a "mask-when-not", D36–D38).** A conservative per-block upper-bound
//! analysis ([`ub_of`]) proves some effective addresses are *already* `< size`; for those
//! the `& mask` is dropped ([`in_window`] / `mask_addr`'s `elide`), since the unmasked
//! address already equals the masked one and stays in-window — closing part of the gap to
//! wasm32's free guard-page accesses. This is the subset of guard-when-bounded that needs no
//! guard region (it only elides *provably in-window* accesses, never relying on a fault); the
//! full wasm32-style large-guard version awaits real guard pages (§5). A wrong bound would be
//! a confinement escape, so the analysis is upper-bound-only (unknown ⇒ mask) and the
//! elision is differentially guarded by the escape-oracle (final-memory equality, §18).
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

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F32, F64, I16, I32, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp as ClifRmwOp, BlockArg, BlockCall, Endianness, Function, InstBuilder,
    JumpTableData, MemFlags, StackSlotData, StackSlotKind, Type, UserFuncName, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use svm_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, ConvOp, Data, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncIdx,
    FuncType, Inst, IntTy, IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator, ValType,
    DEFAULT_RESERVED_LOG2,
};

mod mem; // guest-window allocation + the §4/§5 guard-page / detect-and-kill handler

// JIT fiber runtime (§12): host-side fiber table + `extern "C"` thunks for `cont.new`/`resume`/
// `suspend`, on top of the `svm-fiber` stack-switch substrate. Available where `svm_fiber::supported()`.
#[cfg(fiber_rt)]
mod fiber_rt;

// 1:1 OS-thread executor for `thread.spawn`/`thread.join` + the `wait`/`notify` futex (§12): the VM
// exposes these as *primitives*, not a scheduler — a spawned vCPU is one real OS thread; any M:N model
// is built by the guest runtime over these + `cont.*` (D22: no built-in scheduler). The futex core is
// loom-verified. Available where `svm_fiber::supported()` (x86-64 unix).
#[cfg(fiber_rt)]
mod os_thread_rt;

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
/// `type_id`, then calls `code` — confinement at the use site (invariant I2).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FnEntry {
    type_id: u32,
    _pad: u32,
    code: u64,
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

/// §15 **spawn quota** — host-configurable ceilings on how many fibers (`cont.new`) / vCPUs
/// (`thread.spawn`) a JIT run may create, below the fixed anti-bomb ceilings. The runtimes clamp each
/// to their hard ceiling (a quota only *tightens*); exceeding it is a clean `FiberFault`/`ThreadFault`,
/// matching `svm_interp::Quota`. [`Default`] = the ceilings (an unconfigured run is unchanged). NB the
/// JIT's vCPU table is **cumulative** (a joined slot isn't freed), so `max_vcpus` bounds *total* spawns
/// over the run — stricter than the interpreter's concurrent-liveness cap, but containment holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quota {
    /// Max fibers a run may create (`cont.new`); clamped to the fiber anti-bomb ceiling.
    pub max_fibers: usize,
    /// Max thread cells a run may create (`thread.spawn`); clamped to the vCPU anti-bomb ceiling.
    pub max_vcpus: usize,
}
impl Default for Quota {
    fn default() -> Self {
        Quota {
            max_fibers: 1 << 16,
            max_vcpus: 1 << 16,
        }
    }
}

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

/// The CLIF type backing an IR value type.
fn clif_ty(t: ValType) -> Type {
    match t {
        ValType::I32 => I32,
        ValType::I64 => I64,
        ValType::F32 => F32,
        ValType::F64 => F64,
    }
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
        None, // no kill-path armed (the differential oracle runs to completion)
        None, // no async ring
        None, // no fast cap resolver
        Quota::default(),
    )
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
struct SubWindow {
    base: u64,
    parent_bytes: u64,
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
    interrupt: Option<*const AtomicU64>,
    async_hooks: Option<&dyn AsyncHostHooks>,
    fast_resolver: Option<FastCapResolver>,
    quota: Quota,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let entry = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
    // §5 fuel/epoch kill-path: the address of the host-owned interrupt cell the lowering polls at
    // loop back-edges + function entries. `0` when the caller armed no kill-path (then no checks are
    // emitted — guest code is byte-identical to before). The cell must outlive the run; the caller
    // owns it (e.g. an `Arc<AtomicU64>` a watchdog thread sets), so the baked address stays valid.
    let epoch_addr = interrupt.map_or(0, |p| p as i64);
    // Calls can reach any function, so every function must be lowerable.
    for f in &m.funcs {
        ensure_supported(f)?;
    }

    // Allocate the guest window if the module declares memory: `mapped` backed RW bytes inside
    // a host-configured `reserved` virtual range whose unmapped tail + guard page fault (§4).
    // `mask` is the §4 confinement mask (`reserved − 1`, the mask domain); `win_size` is the
    // seed/snapshot extent (the parent for a sub-window); `cap_mapped` is the child's backed
    // `mapped` that cap-call buffer borrows bound against; `mem_base` is null when none.
    // `sub_base` is the §14 sub-window offset the masking lowering adds (0 for a top-level window).
    let (mut window, mask, win_size, cap_mapped, sub_base): (
        mem::GuestWindow,
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
                        mem::GuestWindow::new(parent, parent),
                        (1u64 << reserved_log2) - 1,
                        parent,
                        mapped as u64,
                        sw.base,
                    )
                }
                None => {
                    let reserved = 1usize << reserved_log2;
                    (
                        mem::GuestWindow::new(mapped, reserved),
                        (1u64 << reserved_log2) - 1,
                        mapped,
                        mapped as u64,
                        0,
                    )
                }
            }
        }
        None => (mem::GuestWindow::new(0, 0), 0, 0, 0, 0),
    };
    // Escape-oracle: seed the window's low bytes so a divergent read/store is observable.
    if let Some(init) = init_mem {
        let n = init.len().min(win_size);
        window.rw_mut()[..n].copy_from_slice(&init[..n]);
    }

    // Initialized data segments (§3a / D40): copy each segment's bytes into the window, then map
    // the `readonly` ones RO (so a guest write to const data faults into the guard, §4/§5). The
    // verifier already bounds every segment to `[0, size)`. Segment offsets are child-relative, so
    // a §14 sub-window shifts them by `sub_base` into the parent backing. Done while fully RW.
    if let Some(mc) = m.memory {
        let size = 1u64 << mc.size_log2;
        let rw = window.rw_mut();
        for d in &m.data {
            let lo = sub_base + d.offset.min(size);
            let hi = sub_base + (d.offset + d.bytes.len() as u64).min(size);
            let (start, end) = (lo as usize, hi as usize);
            rw[start..end].copy_from_slice(&d.bytes[..end - start]);
        }
    }
    for d in &m.data {
        if d.readonly && !d.bytes.is_empty() {
            window.protect_ro(sub_base + d.offset, d.bytes.len() as u64);
        }
    }

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
    // basis for the `call_indirect` type check (matching the interpreter).
    let distinct = distinct_types(&m.funcs);

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
    // Fibers + threads compose via a **per-vCPU** fiber runtime, published through a thread-local. This
    // is the *root* vCPU's runtime (the one running `main` on the caller's thread); each spawned vCPU
    // builds its own from `fiber_cfg` (`os_thread_rt`). Created whenever the module uses `cont.*`.
    #[cfg(fiber_rt)]
    let mut fiber_rt: Option<Box<fiber_rt::FiberRuntime>> = if uses_fibers {
        Some(Box::new(fiber_rt::FiberRuntime::new(
            fiber_type_id,
            fiber_mask,
            quota.max_fibers,
        )))
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
        }
    } else {
        InstEnv::null()
    };
    #[cfg(not(fiber_rt))]
    let inst = InstEnv::null();

    // Define each function body. `clear_context` after each define resets the cached
    // CFG/domtree so the next function never compiles against a stale CFG.
    let mut ctx = module.make_context();
    for (f, id) in m.funcs.iter().zip(&ids) {
        build_clif(
            &mut module,
            &ids,
            &distinct,
            cap,
            fiber,
            thread,
            inst,
            &mut ctx.func,
            f,
            mask,
            cap_mapped,
            sub_base,
            epoch_addr,
        )?;
        module
            .define_function(*id, &mut ctx)
            .map_err(|e| JitError::Backend(e.to_string()))?;
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
    // padded, AoS, host-owned. `call_indirect` masks the guest index into this.
    let table_len = m.funcs.len().next_power_of_two();
    let fn_table: Vec<FnEntry> = (0..table_len)
        .map(|slot| match m.funcs.get(slot) {
            Some(f) => FnEntry {
                type_id: type_id_of(
                    &distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
                _pad: 0,
                code: module.get_finalized_function(ids[slot]) as u64,
            },
            None => FnEntry {
                type_id: PADDING_TYPE_ID,
                _pad: 0,
                code: 0,
            },
        })
        .collect();

    let code = module.get_finalized_function(tramp);
    let mem_base = window.base();
    let mut results = vec![0i64; entry.results.len()];
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
    if let Some(d) = &domain {
        let fiber_cfg = if uses_fibers {
            Some((fiber_type_id, fiber_mask, quota.max_fibers))
        } else {
            None
        };
        d.set_env(
            mem_base as u64,
            fn_table.as_ptr() as u64,
            trap_cell.as_ptr(),
            call_tramp.expect("call-trampoline set for a threaded module"),
            window.fault_range(),
            fiber_cfg,
            epoch_addr as usize, // §5 kill-path: so parked vCPUs (futex/join) observe the interrupt
        );
    }

    // §9/§12 async ring: publish this run's futex-`notify` into the embedder's `Host` so an offload
    // worker can wake a vCPU parked in `submit_async` on a completion counter (the futex `phys` is the
    // parking key). Needs the thread `Domain` (a module that parks on a counter uses `atomic.wait`, so
    // `uses_threads` holds). With no `Domain`/hooks, `submit_async` stays an inert `-EINVAL`.
    #[cfg(fiber_rt)]
    if let (Some(hooks), Some(d)) = (async_hooks, &domain) {
        // The `Domain` pointer as a `usize` so the hook closure is `Send + Sync` (a raw pointer is not,
        // and Rust-2021 disjoint capture would otherwise grab the bare pointer field).
        let dom_addr = (&**d as *const os_thread_rt::Domain) as usize;
        hooks.install_notify(std::sync::Arc::new(move |key: u64, count: u32| {
            let n = count.min(i32::MAX as u32) as i32;
            // SAFETY: the `Domain` outlives the run; the hook is dropped by `hooks.finish()` after
            // `join_all`, before the `Domain` is freed, so the pointer is valid whenever a worker
            // calls this. `thread_notify` is sound from any thread (it locks the domain futex), like a
            // guest `atomic.notify`.
            unsafe { os_thread_rt::thread_notify(dom_addr as *const os_thread_rt::Domain, key, n) };
        }));
    }
    #[cfg(not(fiber_rt))]
    let _ = &async_hooks;

    // §14: the nesting `Nursery`'s address is baked into the Instantiator cap.call sites; a child runs
    // over its own window (allocated per `instantiate`), so there is no per-run env to seed here. Keep
    // it alive until the run completes.
    #[cfg(fiber_rt)]
    let _nursery = &nursery;

    // Publish the root fiber runtime (when the module uses `cont.*`) so its thunks find it via the
    // thread-local for the duration of the entry; spawned vCPUs publish their own.
    #[cfg(fiber_rt)]
    let prev_rt = fiber_rt
        .as_mut()
        .map(|rt| fiber_rt::set_current(&mut **rt as *mut fiber_rt::FiberRuntime));

    // SAFETY: `code` is the finalized entry trampoline honouring the `Entry` ABI. It reads the arg
    // slots, writes the result slots, accesses only the guarded window (any escape faults into the
    // guard page), reads `fn_table`, and writes `trap_cell`. All buffers outlive the call; `module`
    // owns the executable page until dropped below (after every spawned vCPU is joined).
    let faulted = unsafe {
        mem::run_guarded(
            &window,
            code,
            args.as_ptr(),
            results.as_mut_ptr(),
            mem_base,
            fn_table.as_ptr() as *const core::ffi::c_void,
            trap_cell.as_ptr(),
        )
    };
    #[cfg(fiber_rt)]
    if let Some(p) = prev_rt {
        fiber_rt::set_current(p);
    }
    // Join every spawned vCPU OS thread before freeing the window/code — no vCPU may outlive them.
    #[cfg(fiber_rt)]
    if let Some(d) = &domain {
        d.join_all();
    }
    // §9/§12 async ring: now that every vCPU is joined, drain the offload pool and drop the futex hook
    // (which holds the `Domain` pointer) before the window / `Domain` are freed below — so no worker
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
    drop(window);
    drop(fn_table);
    drop(module); // frees the executable memory after the call has returned

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
pub(crate) unsafe fn compile_child_and_run(
    funcs: &[Func],
    child_entry: FuncIdx,
    sub_base: u64,
    child_size_log2: u8,
    parent_mem_base: *mut u8,
    args: &[i64],
    epoch_addr: usize,
) -> Result<(i64, i64), JitError> {
    // The synchronous child's powerbox is empty (an inert `cap.call` → `CapFault`).
    let child = compile_child(
        funcs,
        child_entry,
        child_size_log2,
        empty_cap_thunk,
        core::ptr::null_mut(),
        epoch_addr, // §5 kill-path: the child polls the parent's interrupt cell
    )?;
    let child_size = 1u64 << child_size_log2; // bounded ≤ MAX by compile_child's reject (audit #3)
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

    let mut results = vec![0i64; n_results];
    let mut trap_cell: i64 = 0;
    // SAFETY: `code` honours the `Entry` ABI; it accesses only its own window `[child_base, …+size)`
    // (baked masking; a width-overrun hits this window's guard page), reads the child `fn_table`, and
    // writes its result/trap slots. The guard is re-entrant, so a child fault is caught here and the
    // parent's recovery state is restored.
    let faulted = mem::run_guarded(
        &child_window,
        code,
        args.as_ptr(),
        results.as_mut_ptr(),
        child_base,
        fn_table_ptr as *const core::ffi::c_void,
        &mut trap_cell,
    );
    if faulted {
        trap_cell = mem::FAULT_TRAP;
    }
    // Copy the child's final window back into the parent's sub-region — the parent (the superset) now
    // sees the child's writes (materialized at `instantiate`-completion for a synchronous child). A
    // guest with no Memory cap leaves every page mapped; `restore_rw` is defensive.
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
    Ok((results.first().copied().unwrap_or(0), trap_cell))
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
    /// Owns the executable memory; dropped last.
    module: JITModule,
}

#[cfg(fiber_rt)]
impl Drop for ChildCode {
    fn drop(&mut self) {
        // `JITModule` frees its executable memory on drop; nothing extra to do — this impl exists to
        // document that `code`/`fn_table` die with the struct (no use may outlive it).
        let _ = &self.module;
    }
}

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
        if func_uses_fibers_or_threads(f) {
            return Err(JitError::Unsupported(
                "a §14 JIT child using fibers/threads is not supported yet",
            ));
        }
    }
    let child_size = 1u64 << child_size_log2; // bounded ≤ MAX by compile_child's reject (audit #3)
    let mask = child_size - 1;

    let mut flags = settings::builder();
    let _ = flags.set("is_pic", "false");
    let _ = flags.set("preserve_frame_pointers", "true");
    let _ = flags.set("opt_level", "speed"); // match the top-level compile (GVN/CSE/const-mat)
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
            &distinct,
            cap,
            FiberEnv::null(),
            ThreadEnv::null(),
            InstEnv::null(), // a JIT child cannot itself nest yet (its Instantiator cap.call → CapFault)
            &mut ctx.func,
            f,
            mask,
            child_size,        // the child is fully mapped (reserved == mapped == size)
            0,                 // top-level confinement over the child's own window
            epoch_addr as i64, // §5 kill-path: the child polls the parent's interrupt cell
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
            Some(f) => FnEntry {
                type_id: type_id_of(
                    &distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
                _pad: 0,
                code: module.get_finalized_function(ids[slot]) as u64,
            },
            None => FnEntry {
                type_id: PADDING_TYPE_ID,
                _pad: 0,
                code: 0,
            },
        })
        .collect();

    let code = module.get_finalized_function(tramp);
    Ok(ChildCode {
        fn_table,
        code,
        module,
    })
}

/// Whether `f` uses any §12 fiber/thread/futex op — which a §14 JIT child cannot yet run (it would
/// compile against this run's null fiber/thread thunks). Memory + `cap.call` are fine (the latter is
/// an inert `CapFault` under the child's empty powerbox).
#[cfg(fiber_rt)]
fn func_uses_fibers_or_threads(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::ContNew { .. }
                    | Inst::ContResume { .. }
                    | Inst::Suspend { .. }
                    | Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. }
            )
        })
    })
}

/// The natural CLIF signature for an IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. Both context pointers are threaded through every call so loads/
/// stores reach the window and `call_indirect` reaches the function table.
fn natural_sig(module: &mut JITModule, f: &Func) -> cranelift_codegen::ir::Signature {
    sig_from(module, &f.params, &f.results)
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
    for p in params {
        sig.params.push(AbiParam::new(clif_ty(*p)));
    }
    for r in results {
        sig.returns.push(AbiParam::new(clif_ty(*r)));
    }
    sig
}

/// Reject functions using any op outside the integer slice, so `build_clif` can lower
/// the remainder totally. Keeping the check separate keeps the lowering readable.
fn ensure_supported(f: &Func) -> Result<(), JitError> {
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
                | Inst::FCmp { .. }
                | Inst::FToISat { .. }
                | Inst::FToITrap { .. }
                | Inst::IToFConv { .. }
                | Inst::Cast { .. }
                | Inst::Load { .. }
                | Inst::Store { .. }
                | Inst::AtomicLoad { .. }
                | Inst::AtomicStore { .. }
                | Inst::AtomicRmw { .. }
                | Inst::AtomicCmpxchg { .. }
                | Inst::AtomicFence { .. }
                | Inst::Call { .. }
                | Inst::CallIndirect { .. }
                | Inst::CapCall { .. }
                | Inst::RefFunc { .. }
                | Inst::IntBin { .. }
                | Inst::Convert { .. } => {}
                Inst::IntUn { .. } => {}
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
                    if cfg!(fiber_rt) => {}
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
}

impl FiberEnv {
    fn null() -> FiberEnv {
        FiberEnv {
            new_thunk: 0,
            resume_thunk: 0,
            suspend_thunk: 0,
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
}

impl InstEnv {
    fn null() -> InstEnv {
        InstEnv {
            nursery_addr: 0,
            instantiate_thunk: 0,
            join_thunk: 0,
            coro_spawn_thunk: 0,
            coro_resume_thunk: 0,
        }
    }
    /// True when this compilation may lower `Instantiator` cap.calls to the nesting runtime (the
    /// parent compile with a live `Nursery`); `false` ⇒ they take the ordinary `cap.call` path.
    fn is_active(&self) -> bool {
        self.nursery_addr != 0
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
    /// Address of the host-owned **interrupt cell** (`AtomicU64`) for the §5 fuel/epoch kill-path.
    /// `0` ⇒ no kill-path is armed for this compile (the checks are not emitted — guest code is
    /// byte-identical to the un-armed build). When non-zero, the lowering polls `*epoch_addr` at
    /// loop back-edges and function entries and traps [`TrapKind::OutOfFuel`] if the host has set it
    /// non-zero, so a non-terminating guest is stopped. The guest cannot disable the poll — only the
    /// host (who chose to arm it) writes the cell.
    epoch_addr: i64,
    /// Every function's `FuncId`, so `call`/`return_call` can reference callees.
    ids: &'a [FuncId],
    /// Distinct module signatures, for `call_indirect` type ids.
    distinct: &'a [FuncType],
}

/// Build the natural-ABI CLIF for one IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. The CLIF entry block holds the native params and jumps into IR
/// block 0 passing the parameters as its block args.
#[allow(clippy::too_many_arguments)]
fn build_clif(
    module: &mut JITModule,
    ids: &[FuncId],
    distinct: &[FuncType],
    cap: CapEnv,
    fiber: FiberEnv,
    thread: ThreadEnv,
    inst: InstEnv,
    clif: &mut Function,
    f: &Func,
    mask: u64,
    mapped: u64,
    sub_base: u64,
    epoch_addr: i64,
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
    let entry = b.create_block();
    b.append_block_param(entry, I64); // mem_base
    b.append_block_param(entry, I64); // fn_table_base
    b.append_block_param(entry, I64); // trap_out
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
    let lower = Lower {
        mem_var,
        fn_table_var,
        trap_var,
        result_tys: f.results.iter().map(|t| clif_ty(*t)).collect(),
        mask,
        mapped,
        sub_base,
        fn_table_mask: (ids.len().next_power_of_two() as u64) - 1,
        cap,
        fiber,
        thread,
        inst,
        epoch_addr,
        ids,
        distinct,
    };

    // Jump into IR block 0 passing the function parameters (entry params after the
    // three context pointers). A §5 kill-path check guards the *entry* (caught before any work):
    // this is what stops unbounded recursion and tail-call loops — each (re-)entry polls the
    // interrupt cell. Intra-function loops are caught by the per-back-edge check in `lower_block`.
    let entry_args: Vec<BlockArg> = b.block_params(entry)[3..]
        .iter()
        .map(|v| BlockArg::from(*v))
        .collect();
    emit_epoch_check(&mut b, &lower);
    b.ins().jump(blocks[0], &entry_args);

    for (i, blk) in f.blocks.iter().enumerate() {
        lower_block(module, &mut b, blk, blocks[i], &blocks, &lower)?;
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

    // Decode args (context pointers first), call the entry, store results.
    let mut call_args = vec![mem_base, fn_table_base, trap_out];
    for (i, p) in entry.params.iter().enumerate() {
        let slot = b
            .ins()
            .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        call_args.push(decode_slot(&mut b, slot, *p));
    }
    let callee = module.declare_func_in_func(entry_id, b.func);
    let call = b.ins().call(callee, &call_args);
    let rets: Vec<Value> = b.inst_results(call).to_vec();
    for (i, r) in rets.iter().enumerate() {
        let slot = encode_slot(&mut b, *r);
        b.ins()
            .store(MemFlags::trusted(), slot, results_ptr, (i * 8) as i32);
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
                    Inst::ContNew { .. } | Inst::ContResume { .. } | Inst::Suspend { .. }
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
    for _ in 0..6 {
        clif.signature.params.push(AbiParam::new(I64)); // code, mem_base, fn_table_base, trap_out, sp, arg
    }
    clif.signature.returns.push(AbiParam::new(I64));
    clif.name = UserFuncName::user(0, 2);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    for _ in 0..6 {
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(blk);
    b.seal_block(blk);
    let p = b.block_params(blk).to_vec();
    let (code, mem_base, fn_table_base, trap_out, sp, arg) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    // The guest entry's natural Tail signature: (mem_base, fn_table_base, trap_out, sp, arg) -> i64.
    let sig = b.import_signature(sig_from(
        module,
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
    ));
    let call = b
        .ins()
        .call_indirect(sig, code, &[mem_base, fn_table_base, trap_out, sp, arg]);
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

    for inst in &blk.insts {
        // `call`/`call_indirect` append 0..N results — handle before the single-value
        // match (which produces exactly one value).
        if let Inst::Call { func, args } = inst {
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = ctx_args(b, lower);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call(callee, &cargs);
            // A trap raised inside the callee leaves the trap cell set and returns zeros; propagate
            // it here so it unwinds immediately (else the caller would run on with bogus results,
            // and a later successful `cap.call` could reset the cell, masking the trap).
            emit_trap_propagate(b, lower);
            vals.extend_from_slice(b.inst_results(call));
            ubs.resize(vals.len(), UB_TOP); // call results are unknown
            continue;
        }
        if let Inst::CallIndirect { ty, idx, args } = inst {
            let code = indirect_dispatch(b, lower, get(&vals, *idx)?, ty);
            let sig = b.import_signature(sig_from(module, &ty.params, &ty.results));
            let mut cargs = ctx_args(b, lower);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call_indirect(sig, code, &cargs);
            // Propagate a callee trap immediately (see the direct-call case above).
            emit_trap_propagate(b, lower);
            vals.extend_from_slice(b.inst_results(call));
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
                lower_instantiator(module, b, lower, *op, *handle, args, &mut vals)?;
            } else if let Some(target) = fast_cap_target(lower, *type_id, *op, sig) {
                // D45 devirtualized fast path: a register-to-register direct call to the specialized
                // host fn the resolver claimed for this `(type_id, op)`.
                lower_cap_call_fast(module, b, lower, target, sig, *handle, args, &mut vals)?;
            } else {
                lower_cap_call(
                    module, b, lower, *type_id, *op, sig, *handle, args, &mut vals,
                )?;
            }
            ubs.resize(vals.len(), UB_TOP); // cap-call results are unknown
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
            tsig.returns.push(AbiParam::new(I32));
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
            // fiber_resume(handle:i32, arg:i64, status_out:*i64, trap_out:i64) -> value:i64.
            // Results are appended (status:i32, value:i64) to match the IR's two-result shape.
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let kh = get(&vals, *k)?;
            let av = get(&vals, *arg)?;
            let trap_out = b.use_var(lower.trap_var);
            let mut tsig = module.make_signature();
            for t in [I32, I64, I64, I64] {
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
            // thread_wait(sched, phys:i64, expected:i64, width:i32, timeout:i64) -> status:i32
            let w = atomic_width(*ty);
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false);
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
            let mut tsig = module.make_signature();
            for t in [I64, I64, I64, I32, I64] {
                tsig.params.push(AbiParam::new(t));
            }
            tsig.returns.push(AbiParam::new(I32));
            let tref = b.import_signature(tsig);
            let thunk = b.ins().iconst(I64, lower.thread.wait_thunk);
            let call = b
                .ins()
                .call_indirect(tref, thunk, &[sched, phys, exp, width, to]);
            vals.push(b.inst_results(call)[0]);
            ubs.resize(vals.len(), UB_TOP);
            continue;
        }
        if let Inst::MemoryNotify { addr, count } = inst {
            // thread_notify(sched, phys:i64, count:i32) -> woken:i32. Accesses no memory (the address
            // is only confined, no alignment requirement — matching the interpreter).
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false);
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
            Inst::Load {
                op, addr, offset, ..
            } => {
                let elide = in_window(ub_at(&ubs, *addr), *offset, op.info().2, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
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
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
                lower_store(b, *op, phys, get(&vals, *value)?);
                continue; // store produces no value
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
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
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
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
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
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
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
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide);
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
            // Natural ABI: return the result values directly (CLIF multi-return).
            let rets: Vec<Value> = outs
                .iter()
                .map(|o| get(&vals, *o))
                .collect::<Result<_, _>>()?;
            b.ins().return_(&rets);
        }
        Terminator::ReturnCall { func, args } => {
            // Tail call (§3b): replace this frame with the callee, threading the context.
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = ctx_args(b, lower);
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
/// fn_table_base, trap_out)`.
fn ctx_args(b: &mut FunctionBuilder, lower: &Lower) -> Vec<Value> {
    vec![
        b.use_var(lower.mem_var),
        b.use_var(lower.fn_table_var),
        b.use_var(lower.trap_var),
    ]
}

/// Lower a trap (§5 detect-and-kill): store the kind code into the host trap cell, then
/// `return` dummy zero results so the run unwinds to the trampoline, which reports the
/// trap. (The reference JIT detects traps this way; production uses hardware faults.)
///
/// Caveat: this returns from the *current* function only. The current scalar tests put
/// every trap in the entry function (or its dispatch), so that suffices; propagating a
/// trap *out of a callee* would need a post-call check, added when a case needs it.
fn emit_trap(b: &mut FunctionBuilder, lower: &Lower, kind: TrapKind) {
    let cell = b.use_var(lower.trap_var);
    let code = b.ins().iconst(I64, kind as u32 as i64); // full i64 cell (high bits 0)
    b.ins().store(MemFlags::trusted(), code, cell, 0);
    let zeros: Vec<Value> = lower.result_tys.iter().map(|t| zero_of(b, *t)).collect();
    b.ins().return_(&zeros);
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

/// A zero constant of CLIF type `t` (for a trapping path's dummy return).
fn zero_of(b: &mut FunctionBuilder, t: Type) -> Value {
    if t == F32 {
        b.ins().f32const(0.0)
    } else if t == F64 {
        b.ins().f64const(0.0)
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
    let zeros: Vec<Value> = lower.result_tys.iter().map(|t| zero_of(b, *t)).collect();
    b.ins().return_(&zeros);
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
    handle: u32,
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
    let h = get(vals, handle)?;
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
fn lower_instantiator(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    op: u32,
    handle: u32,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    let nursery = b.ins().iconst(I64, lower.inst.nursery_addr);
    let mem_base = b.use_var(lower.mem_var);
    let trap_out = b.use_var(lower.trap_var);
    match op {
        0 | 5 => {
            // instantiate(nursery, mem_base, handle:i32, module:i64, entry:i64, off:i64,
            //             size_log2:i64, fuel:i64, trap_out:i64) -> child_handle:i32. op 0 is a
            // **self** child (module = -1); op 5 (`instantiate_module`, §14 separate-module child)
            // passes a host-granted `Module` handle as its first arg and shifts the rest by one.
            let h = get(vals, handle)?; // the Instantiator handle (resolved for authority)
            let (modh, a0) = if op == 5 {
                (get(vals, *args.first().ok_or(JitError::Malformed)?)?, 1)
            } else {
                (b.ins().iconst(I64, -1), 0)
            };
            let entry = get(vals, *args.get(a0).ok_or(JitError::Malformed)?)?;
            let off = get(vals, *args.get(a0 + 1).ok_or(JitError::Malformed)?)?;
            let size_log2 = get(vals, *args.get(a0 + 2).ok_or(JitError::Malformed)?)?;
            let fuel = get(vals, *args.get(a0 + 3).ok_or(JitError::Malformed)?)?;
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
            vals.push(b.inst_results(call)[0]);
        }
        1 => {
            // join(nursery, child_handle:i32, trap_out:i64) -> result:i64. The cap.call's handle
            // operand (the Instantiator) is unused here — the child handle is the first arg, and the
            // nursery owns the child table for this run.
            let child = get(vals, *args.first().ok_or(JitError::Malformed)?)?;
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
            vals.push(b.inst_results(call)[0]);
        }
        2 | 4 | 6 | 7 => {
            // coro_spawn(nursery, mem_base, handle:i32, module:i64, entry:i64, off:i64,
            //            size_log2:i64, fuel:i64, demand:i32, trap_out:i64) -> child_handle:i32 —
            // §14 co-fiber spawn. ops 2/4 are **self** children (module = -1); ops 6/7
            // (`spawn[_demand]_coroutine_module`) pass a `Module` handle first and shift the rest.
            // ops 4/7 demand-page the child's window for fault-driven yield.
            let h = get(vals, handle)?;
            let (modh, a0) = if op >= 6 {
                (get(vals, *args.first().ok_or(JitError::Malformed)?)?, 1)
            } else {
                (b.ins().iconst(I64, -1), 0)
            };
            let entry = get(vals, *args.get(a0).ok_or(JitError::Malformed)?)?;
            let off = get(vals, *args.get(a0 + 1).ok_or(JitError::Malformed)?)?;
            let size_log2 = get(vals, *args.get(a0 + 2).ok_or(JitError::Malformed)?)?;
            let fuel = get(vals, *args.get(a0 + 3).ok_or(JitError::Malformed)?)?;
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
            vals.push(b.inst_results(call)[0]);
        }
        3 => {
            // coro_resume(nursery, mem_base, handle:i32, child:i32, value:i64, status_out:*i64,
            //             trap_out:i64) -> value:i64. Results are appended `(status:i32, value:i64)`
            // to match the op's two-result shape (like `cont.resume`).
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let h = get(vals, handle)?;
            let child = get(vals, *args.first().ok_or(JitError::Malformed)?)?;
            let value = get(vals, *args.get(1).ok_or(JitError::Malformed)?)?;
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
            let status = b.ins().ireduce(I32, status64);
            vals.push(status);
            vals.push(value_out);
        }
        _ => return Err(JitError::Unsupported("unknown Instantiator op")),
    }
    Ok(())
}

fn get(vals: &[Value], i: u32) -> Result<Value, JitError> {
    vals.get(i as usize).copied().ok_or(JitError::Malformed)
}

/// The §4 confinement masking lowering (invariant I1): compute the physical address
/// `mem_base + sub_base + ((addr + offset) & mask)`. The `sub_base + ((addr + offset) & mask)`
/// is exactly `svm_mask::Window::sub(...).confine`, so the JIT and the isolated masking unit
/// agree — and for a top-level window (`sub_base == 0`) it collapses to the plain
/// `mem_base + ((addr + offset) & mask)` of [`svm_mask::Window::confine`].
///
/// When `elide` is set the `& mask` is dropped — but **only** the caller's
/// [`in_window`] proof (the address is provably `< size`) may set it, so the unmasked
/// `addr + offset` already equals the masked value and stays in `[0, size)`. This is the
/// "mask-when-not" / elide-when-provably-bounded half of §1a (D36–D38); a wrong proof is a
/// confinement escape, caught by the escape-oracle (final-memory differential, §18). The
/// `+ sub_base` is independent of elision (it shifts the whole `[0, size)` child window into
/// its parent slice) and is itself elided when `sub_base == 0`.
fn mask_addr(
    b: &mut FunctionBuilder,
    lower: &Lower,
    addr: Value,
    offset: u64,
    elide: bool,
) -> Value {
    // Fold the immediate only when non-zero, so an offset-0 access keeps a minimal address
    // expression (helps Cranelift's GVN / store-to-load forwarding recognize equal addresses).
    let eff = if offset == 0 {
        addr
    } else {
        let off = b.ins().iconst(I64, offset as i64);
        b.ins().iadd(addr, off)
    };
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
        ValType::I64 => slot,
        ValType::I32 => b.ins().ireduce(I32, slot),
        ValType::F32 => {
            let i = b.ins().ireduce(I32, slot);
            b.ins().bitcast(F32, MemFlags::new(), i)
        }
        ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), slot),
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

fn float_bin(b: &mut FunctionBuilder, op: FBinOp, x: Value, y: Value) -> Value {
    match op {
        FBinOp::Add => b.ins().fadd(x, y),
        FBinOp::Sub => b.ins().fsub(x, y),
        FBinOp::Mul => b.ins().fmul(x, y),
        FBinOp::Div => b.ins().fdiv(x, y),
        FBinOp::Min => b.ins().fmin(x, y),
        FBinOp::Max => b.ins().fmax(x, y),
        FBinOp::Copysign => b.ins().fcopysign(x, y),
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
