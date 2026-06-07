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
    AtomicRmwOp, BinOp, Block, CastOp, ConvOp, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncIdx,
    FuncType, Inst, IntTy, IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator, ValType,
    DEFAULT_RESERVED_LOG2,
};

mod mem; // guest-window allocation + the §4/§5 guard-page / detect-and-kill handler

// The windows placeholder-window commit primitive, reused by `svm-run`'s Memory-cap backend (it
// commits/grows tail pages of this same window; a plain `VirtualAlloc(MEM_COMMIT)` cannot commit a
// placeholder reservation). See `mem::win_commit_rw`.
#[cfg(windows)]
pub use mem::win_commit_rw;

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
struct FnEntry {
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
    )
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
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let entry = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
    // Calls can reach any function, so every function must be lowerable.
    for f in &m.funcs {
        ensure_supported(f)?;
    }

    // Allocate the guest window if the module declares memory: `mapped` backed RW bytes inside
    // a host-configured `reserved` virtual range whose unmapped tail + guard page fault (§4).
    // `mask` is the §4 confinement mask (`reserved − 1`, the mask domain); `win_size` is the
    // backed `mapped` extent (what we seed/snapshot); `mem_base` is null when none.
    let (mut window, mask, win_size): (mem::GuestWindow, u64, usize) = match m.memory {
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
            let reserved = 1usize << reserved_log2;
            (
                mem::GuestWindow::new(mapped, reserved),
                (1u64 << reserved_log2) - 1,
                mapped,
            )
        }
        None => (mem::GuestWindow::new(0, 0), 0, 0),
    };
    // Escape-oracle: seed the window's low bytes so a divergent read/store is observable.
    if let Some(init) = init_mem {
        let n = init.len().min(win_size);
        window.rw_mut()[..n].copy_from_slice(&init[..n]);
    }

    // Initialized data segments (§3a / D40): copy each segment's bytes into the window, then map
    // the `readonly` ones RO (so a guest write to const data faults into the guard, §4/§5). The
    // verifier already bounds every segment to `[0, size)`. Done while the window is fully RW.
    if let Some(mc) = m.memory {
        let size = 1u64 << mc.size_log2;
        let rw = window.rw_mut();
        for d in &m.data {
            let end = (d.offset + d.bytes.len() as u64).min(size) as usize;
            let start = (d.offset as usize).min(end);
            rw[start..end].copy_from_slice(&d.bytes[..end - start]);
        }
    }
    for d in &m.data {
        if d.readonly && !d.bytes.is_empty() {
            window.protect_ro(d.offset, d.bytes.len() as u64);
        }
    }

    let mut flags = settings::builder();
    // A JIT'd function is called directly, not relocated into a shared object.
    let _ = flags.set("is_pic", "false");
    // Cranelift's x64 `return_call` (tail calls, §3b) lowering requires frame pointers.
    let _ = flags.set("preserve_frame_pointers", "true");
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
    };

    // Define each function body. `clear_context` after each define resets the cached
    // CFG/domtree so the next function never compiles against a stale CFG.
    let mut ctx = module.make_context();
    for (f, id) in m.funcs.iter().zip(&ids) {
        build_clif(
            &mut module,
            &ids,
            &distinct,
            cap,
            &mut ctx.func,
            f,
            mask,
            win_size as u64,
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
    module
        .finalize_definitions()
        .map_err(|e| JitError::Backend(e.to_string()))?;

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
    // code for an Exit. A trapping path (or the cap thunk) writes it.
    let mut trap_cell: i64 = 0;
    // SAFETY: `code` is the finalized trampoline with the entry signature
    // (`build_trampoline` set it). It reads `entry.params.len()` arg slots, writes
    // `entry.results.len()` result slots, accesses only the guarded window (masking
    // confines effective addresses; any escape faults into the guard page), reads
    // `fn_table` (length `table_len`, masked index), and writes `trap_cell`. All buffers
    // outlive the call; `module` owns the executable page until dropped below.
    let faulted = unsafe {
        mem::run_guarded(
            &window,
            code,
            args.as_ptr(),
            results.as_mut_ptr(),
            mem_base,
            fn_table.as_ptr() as *const core::ffi::c_void,
            &mut trap_cell,
        )
    };
    // A caught guard fault is detect-and-kill (§5): report MemoryFault to the host.
    if faulted {
        trap_cell = mem::FAULT_TRAP;
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

    let code = trap_cell as u32;
    let outcome = if code == 0 {
        JitOutcome::Returned(results)
    } else if code == EXIT_CODE {
        JitOutcome::Exited((trap_cell >> 32) as i32)
    } else {
        JitOutcome::Trapped(TrapKind::from_code(code).ok_or(JitError::Malformed)?)
    };
    Ok((outcome, final_mem))
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
                | Inst::Call { .. }
                | Inst::CallIndirect { .. }
                | Inst::CapCall { .. }
                | Inst::RefFunc { .. }
                | Inst::IntBin { .. }
                | Inst::Convert { .. } => {}
                Inst::IntUn { op, .. } => match op {
                    IntUnOp::Clz | IntUnOp::Ctz | IntUnOp::Popcnt => {}
                    _ => return Err(JitError::Unsupported("int extend ops")),
                },
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
    /// The function-table index mask (`next_pow2(nfuncs) - 1`) for `call_indirect`.
    fn_table_mask: u64,
    /// The host `cap.call` thunk + ctx (constant addresses).
    cap: CapEnv,
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
    clif: &mut Function,
    f: &Func,
    mask: u64,
    mapped: u64,
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
        fn_table_mask: (ids.len().next_power_of_two() as u64) - 1,
        cap,
        ids,
        distinct,
    };

    // Jump into IR block 0 passing the function parameters (entry params after the
    // three context pointers).
    let entry_args: Vec<BlockArg> = b.block_params(entry)[3..]
        .iter()
        .map(|v| BlockArg::from(*v))
        .collect();
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
            lower_cap_call(
                module, b, lower, *type_id, *op, sig, *handle, args, &mut vals,
            )?;
            ubs.resize(vals.len(), UB_TOP); // cap-call results are unknown
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
            Inst::IntUn { op, a, .. } => {
                let x = get(&vals, *a)?;
                match op {
                    IntUnOp::Clz => b.ins().clz(x),
                    IntUnOp::Ctz => b.ins().ctz(x),
                    IntUnOp::Popcnt => b.ins().popcnt(x),
                    _ => return Err(JitError::Unsupported("int extend ops")),
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
            b.ins().brif(c, tb, &ta, eb, &ea);
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let index = get(&vals, *idx)?;
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

fn get(vals: &[Value], i: u32) -> Result<Value, JitError> {
    vals.get(i as usize).copied().ok_or(JitError::Malformed)
}

/// The §4 confinement masking lowering (invariant I1): compute the physical address
/// `mem_base + ((addr + offset) & mask)`. The `(addr + offset) & mask` is exactly
/// `svm_mask::Window::confine`, so the JIT and the isolated masking unit agree.
///
/// When `elide` is set the `& mask` is dropped — but **only** the caller's
/// [`in_window`] proof (the address is provably `< size`) may set it, so the unmasked
/// `addr + offset` already equals the masked value and stays in `[0, size)`. This is the
/// "mask-when-not" / elide-when-provably-bounded half of §1a (D36–D38); a wrong proof is a
/// confinement escape, caught by the escape-oracle (final-memory differential, §18).
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
