//! Phase-1b bytecode engine (see `INTERP_PERF.md`).
//!
//! Compiles a function once into a flat, operand-resolved op stream over a **function-wide
//! global-slot register file**, executed with **register windows** for calls (each activation
//! occupies `[base, base + nslots)` of one shared `regs` vector — a call opens the next window with
//! no per-call allocation, a return writes results back and restores the caller's window). This is
//! the production form of the Phase-1 ROI spike; it reuses the crate's audited semantic helpers
//! (`bin64`, `cmp32`, `fto_i`, …) and `Mem` — **no op semantics are duplicated here**, only the
//! dispatch/layout.
//!
//! Scope so far: scalar + memory + SIMD/`v128` + fences + direct & indirect calls; the synchronous
//! capability seam (generic `cap.call` + `cap.self.*`, via `host.cap_dispatch_slots`); §12 **fibers**
//! (`cont.*`/`suspend`, cooperative single-vCPU switching in [`step_vcpu`]); and §12 **threads**
//! (`thread.spawn`/`join` + `memory.wait`/`notify`) on a cooperative single-threaded scheduler
//! ([`drive`]) over one shared `Mem`; and §14 **coroutines** (`Instantiator.spawn_coroutine`/`resume`
//! + `Yielder.yield`, inline-driven over a confined `nested_view` child window — including the
//! separate-**module** and **demand** (fault-driven-yield, lazy-paged) variants) and §14 **executor
//! children** (`Instantiator.instantiate`/`join` + the separate-module variant, scheduler-driven over
//! a confined child env with an attenuated `Instantiator`+`AddressSpace` powerbox and a `quota`
//! sub-budget) — §14 is fully covered (ops 0–7). Faithful for the
//! interleaving-invariant programs the oracle uses; and §22 **guest-driven JIT units**
//! (`Jit.install`/`uninstall`/`invoke` + cross-module `call_indirect` into an installed unit) over a
//! multi-module [`Domain`] (a runtime dispatch table spanning `mods`; `invoke` runs a unit nested
//! over the shared window/table). Hot scalar/memory ops dispatch inline; the SIMD/`v128`/fence long
//! tail is delegated to the reference [`super::eval_inst`]. Threads and fibers compose (the fiber
//! registry is run-shared, so fibers migrate across vCPUs); and **tail calls** (`return_call`/
//! `return_call_indirect`, reusing the current window — O(1) deep tail recursion); §GC **`gc.roots`**
//! (conservative root enumeration over the whole vCPU continuation — sound, not bit-identical, per
//! GC.md §3.2); and **durability** freeze/thaw for single-fiber vCPUs (IR-driven by the `svm-durable`
//! transform — the engine just runs the transformed module over a seeded window, via
//! [`compile_and_run_capture_reserved_with_host`]). [`compile_module`] returns `None` when a function
//! needs a seam not yet driven here — instantiate-mixed-with-fibers, `gc.roots`-mixed-with-threads, or
//! **multi-fiber** durable freeze — so callers (`super::run_with_host_fast`) fall back to the
//! tree-walker for those.
//!
//! `run`/`run_with_host` stay the tree-walker (the reference oracle); the bytecode engine is reached
//! via `run_fast`/`run_with_host_fast` (and, with a trap-time backtrace, `run_with_host_fast_traced`).
//! Correctness is gated by exact-equality harnesses against the tree-walker (`bytecode_diff.rs` — which
//! also checks trap-backtrace parity on every trapping generated module, `bytecode_{caps,fibers,threads,
//! coroutines,instantiate,separate_module,demand_coroutine,tailcall,debug,traced,gc_roots,durable,
//! dynlink}.rs`; `gc_roots` checks soundness rather than equality; `durable` checks freeze/thaw artifact
//! + round-trip equality; `traced` checks trap-time backtrace `IrPc`-equality with `run_with_host_traced`).
//!
//! Like the reference interpreter, it is total and panic-free: every slot/pc index is in range by
//! construction of the compiler, and `compile_module` rejects anything it can't lower.

use svm_ir::{
    BinOp, CastOp, CmpOp, ConvOp, DebugInfo, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx,
    IToF, Inst, IntTy, IntUnOp, LoadOp, Module, SpawnRec, StoreOp, Terminator, ValType, VarLoc,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, slot_to_val, step, trunc_trap, val_to_slot, GuestMem, Host, LockUnpoisoned,
    Mem, Reg, Trap, Value, VarValue, DEFAULT_RESERVED_LOG2,
};

// ---- Per-function call profiler (opt-in `callprof` feature; tier-up break-even measurement) -------
// A thread-local histogram indexed by primary-module function index, bumped once per `Op::Call`. Off
// by default: with the feature disabled these items don't exist and the hot path is byte-identical.
#[cfg(feature = "callprof")]
mod callprof {
    use std::cell::RefCell;
    thread_local! {
        static COUNTS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    }
    /// Arm the profiler with a zeroed histogram of `n` functions (call before the run).
    pub fn reset(n: usize) {
        COUNTS.with(|c| {
            let mut c = c.borrow_mut();
            c.clear();
            c.resize(n, 0);
        });
    }
    /// Record one call to `func` (a no-op if the histogram is smaller — non-primary-module callees).
    #[inline]
    pub fn hit(func: usize) {
        COUNTS.with(|c| {
            if let Some(slot) = c.borrow_mut().get_mut(func) {
                *slot += 1;
            }
        });
    }
    /// Snapshot the per-function call counts.
    pub fn snapshot() -> Vec<u64> {
        COUNTS.with(|c| c.borrow().clone())
    }
}
/// Arm the per-function call profiler with a zeroed `n`-function histogram (opt-in `callprof`).
#[cfg(feature = "callprof")]
pub fn callprof_reset(n: usize) {
    callprof::reset(n);
}
/// Snapshot per-function call counts since the last [`callprof_reset`].
#[cfg(feature = "callprof")]
pub fn callprof_snapshot() -> Vec<u64> {
    callprof::snapshot()
}

/// Block-argument moves applied on a taken edge: `(src_slot, dst_slot)` pairs (frame-relative), with
/// a precomputed `aliasing` flag. A **non-aliasing** edge — the common case (an induction variable /
/// accumulator reads distinct value slots and writes the successor's param slots) — is applied by a
/// single direct pass. An **aliasing** edge, where some destination slot is also read as a source (a
/// parallel move that permutes/swaps params), must gather into `scratch` then scatter, so a value
/// isn't clobbered before it is read. Classifying once at compile time keeps the value off the hot
/// path (the `scratch` `push`+read per copy is only paid where correctness needs it).
struct Copies {
    pairs: Box<[(u32, u32)]>,
    aliasing: bool,
}
impl Copies {
    /// Build from resolved `(src, dst)` pairs, classifying `aliasing` = some `dst` is also some `src`
    /// (so a one-pass sequential copy could clobber a not-yet-read source). O(n²) over a tiny edge.
    fn new(pairs: Box<[(u32, u32)]>) -> Copies {
        let aliasing = pairs
            .iter()
            .any(|&(_, d)| pairs.iter().any(|&(s, _)| s == d));
        Copies { pairs, aliasing }
    }
}
/// A resolved branch edge: its arg copies plus the target op index (`pc`).
type Edge = (Copies, u32);

/// One resolved operation. Operands and results are **frame-window-relative slot indices** (added
/// to the activation's `base` at run time); branch targets are op indices (`pc`) within the same
/// function. Edge copies are `(src_slot, dst_slot)` pairs applied on a taken branch.
enum Op {
    Const {
        dst: u32,
        val: Reg,
    },
    IntBin {
        dst: u32,
        a: u32,
        b: u32,
        ty: IntTy,
        op: BinOp,
    },
    IntCmp {
        dst: u32,
        a: u32,
        b: u32,
        ty: IntTy,
        op: CmpOp,
    },
    IntUn {
        dst: u32,
        a: u32,
        ty: IntTy,
        op: IntUnOp,
    },
    Eqz {
        dst: u32,
        a: u32,
        ty: IntTy,
    },
    Convert {
        dst: u32,
        a: u32,
        op: ConvOp,
    },
    Select {
        dst: u32,
        cond: u32,
        a: u32,
        b: u32,
    },
    FBin {
        dst: u32,
        a: u32,
        b: u32,
        ty: FloatTy,
        op: FBinOp,
    },
    FUn {
        dst: u32,
        a: u32,
        ty: FloatTy,
        op: FUnOp,
    },
    FCmp {
        dst: u32,
        a: u32,
        b: u32,
        ty: FloatTy,
        op: FCmpOp,
    },
    FToISat {
        dst: u32,
        a: u32,
        op: FToI,
    },
    FToITrap {
        dst: u32,
        a: u32,
        op: FToI,
    },
    IToFConv {
        dst: u32,
        a: u32,
        op: IToF,
    },
    Cast {
        dst: u32,
        a: u32,
        op: CastOp,
    },
    RefFunc {
        dst: u32,
        func: u32,
    },
    Load {
        dst: u32,
        addr: u32,
        op: LoadOp,
        offset: u64,
    },
    Store {
        addr: u32,
        value: u32,
        op: StoreOp,
        offset: u64,
    },
    // Bulk-memory ops (D62). `MemCopy`/`MemMove` share the overlap-safe `Mem::mem_copy`.
    MemCopy {
        dst: u32,
        src: u32,
        len: u32,
    },
    MemMove {
        dst: u32,
        src: u32,
        len: u32,
    },
    MemFill {
        dst: u32,
        val: u32,
        len: u32,
    },
    AtomicLoad {
        dst: u32,
        addr: u32,
        ty: IntTy,
        offset: u64,
    },
    AtomicStore {
        addr: u32,
        value: u32,
        ty: IntTy,
        offset: u64,
    },
    AtomicRmw {
        dst: u32,
        addr: u32,
        value: u32,
        ty: IntTy,
        op: svm_ir::AtomicRmwOp,
        offset: u64,
    },
    AtomicCmpxchg {
        dst: u32,
        addr: u32,
        expected: u32,
        replacement: u32,
        ty: IntTy,
        offset: u64,
    },
    Br {
        copies: Copies,
        target: u32,
    },
    BrIf {
        cond: u32,
        then_copies: Copies,
        then_pc: u32,
        else_copies: Copies,
        else_pc: u32,
    },
    /// Slice 5a superinstruction: a block-final `IntCmp` fused with the `BrIf` that is its sole
    /// consumer — compare `a`/`b` (`ty`, `op`) and branch on the result, dropping one dispatch plus
    /// the boolean's write-then-reread. Emitted only by the **fused** compile (the fast path); the
    /// debug/trace compile is unfused so its step trace keeps one location per source instruction.
    BrIfCmp {
        a: u32,
        b: u32,
        ty: IntTy,
        op: CmpOp,
        then_copies: Copies,
        then_pc: u32,
        else_copies: Copies,
        else_pc: u32,
    },
    BrTable {
        idx: u32,
        arms: Box<[Edge]>,
        default: Edge,
    },
    Call {
        callee: u32,
        args: Box<[u32]>,
        dst: u32,
    },
    /// `call_indirect` through module 0's natural function table (slot `i` ⇒ func `i`; padding to a
    /// power of two traps). Resolved at run time from `idx` masked to the table length, then the
    /// resolved function's signature is checked against `want_params`/`want_results` (a forged or
    /// mistyped slot is an inert [`Trap::IndirectCallType`], matching [`super::dispatch_indirect`]).
    CallIndirect {
        idx: u32,
        args: Box<[u32]>,
        dst: u32,
        want_params: Box<[ValType]>,
        want_results: Box<[ValType]>,
    },
    /// Synchronous capability call (§3c) through the host powerbox — the guest is suspended, the
    /// host computes a result, and execution continues in the same activation (no scheduler/fiber).
    /// Only the **generic** powerbox path is lowered here; the executor/fiber capability variants
    /// (`Instantiator`, `Yielder`, `JIT`, `SharedRegion` op 4) are rejected by [`compile_inst`] and
    /// fall back to the tree-walker. Args/results cross as `i64` slots (the host-dispatch ABI);
    /// `results` carries `sig.results` so each returned slot is re-typed exactly as the tree-walker
    /// does.
    CapCall {
        type_id: u32,
        op: u32,
        handle: u32,
        args: Box<[u32]>,
        dst: u32,
        /// The call's `sig.params` — carried so an **import-bound** call that resolves to a §22 `Jit`
        /// driver op (`invoke`/`install`/`uninstall`) can be marshalled to the driver like a static
        /// `cap.call (JIT, op)` (which lowers straight to [`Op::JitInvoke`]). Empty when unused.
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §3.6 serve-loop core (ISSUES.md I36 slice 1): `svc.poll` (`cap.call CAP_SELF 9`) — drain
    /// the domain's inbound queue, running each servable dispatch as a handler activation over
    /// the one world. Rewind-driven like the tree-walk serve arm: an admitted handler's return
    /// linkage re-enters THIS op (pc un-advanced) with its result in `dst` (the linkage's result
    /// slot), which the re-execution settles into the ticket's completion cell before admitting
    /// the next dispatch; the final execution overwrites `dst` with the served count. Compiled
    /// only when the module-level qualification veto admits it (no park-capable seams — see
    /// [`compile_module`]), so a handler always runs to completion or traps.
    SvcPoll {
        dst: u32,
        /// `svc.wait` (op 10): identical drain, but a no-progress empty-queue execution parks
        /// the task on its domain ([`Outcome::SvcWait`]) instead of delivering a zero count; a
        /// caller's enqueue re-admits it and the rewound op re-executes the whole drain.
        wait: bool,
    },
    /// FORK.md §9.2 — `clone_caller` (self-op 11): fork-returns-twice, servicer side. Compiled only
    /// in a fork-serving module ([`Seams::bytecode_serves_fork`]). From within a serve handler, it
    /// duplicates the caller parked on this dispatch into a live twin and replies differently to each.
    /// The twin build needs the driver's task/env set, so the op resolves its reply args here and
    /// surfaces to the cooperative driver ([`Outcome::CloneCaller`]); the driver reads the running
    /// handler's `serve_ticket` to name the parked caller. Arity picks the mode (mirrors the oracle):
    /// 2 args = explicit `(reply_orig, reply_twin)`; 0/1 args = **pid mode** (`fork()`) — the parent
    /// sees the twin's task id, the child sees the arg (0). `-EINVAL` outside a handler.
    CloneCaller {
        /// The reply-value arg registers (0, 1, or 2), resolved to i64s in the driver.
        args: Box<[u32]>,
        dst: u32,
        /// Whether the `cap.call` has a result slot (the twin handle / errno lands here).
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap` (self-op 12): the servicer side of `wait(pid)`. From within a serve
    /// handler, reap a twin `pid` a prior `clone_caller` minted, on behalf of the caller parked on
    /// this dispatch — delivering the twin's exit status (now, or when it finishes). Surfaces to the
    /// driver ([`Outcome::Reap`]), which owns the task set + the `forked_twins` allow-set. `-EINVAL`
    /// outside a handler; `-ECHILD` for a `pid` this servicer did not mint (never a hang).
    Reap {
        /// The `pid` arg register (`None` = no arg → an out-of-range pid → `-ECHILD`).
        pid: Option<u32>,
        dst: u32,
        has_result: bool,
    },
    /// §3.6 (I36 slice 2) — `Instantiator.child_offer` (op 14): mint a live-callee offer over a
    /// running child's impl-export into the wirer's table. The authority check (the Instantiator
    /// handle) runs in the op exec; the mint itself needs the child's env/host, so it surfaces to
    /// the driver ([`Outcome::ChildOffer`]).
    ChildOffer {
        handle: u32,
        child: u32,
        export: u32,
        dst: u32,
    },
    // §7/§6 capability reflection `cap.self.count`/`get`/`resolve`/`label`/`attest` are no longer
    // dedicated bytecode ops — they arrive as `cap.call CAP_SELF op 0/1/2/3/4` and compile to the
    // generic `Op::CapCall` (host `cap_dispatch_slots`), the same path the JIT thunk takes.
    /// §3.5 self-namespace extensions through the shared dispatch entry: `op` packs
    /// `(selfop | idx << 8)` (6 = `cap.self.type_id`, 7 = `cap.self.covers`, 8 =
    /// `export.handle`); `handle` is the optional live handle-register (covers only). One
    /// `i32` result.
    CapSelfExt {
        op: u32,
        handle: Option<u32>,
        dst: u32,
    },
    /// §12 fiber create (`cont.new`): register a pending fiber `(funcref, sp)` in the driver's
    /// registry and write its handle to `dst`. No switch — handled by the driver.
    ContNew {
        func: u32,
        sp: u32,
        dst: u32,
    },
    /// §12 fiber resume (`cont.resume`): switch into fiber `k`, delivering `arg`; the two results
    /// `(status, value)` land in `dst`, `dst+1` when the fiber suspends or returns. Driver-driven.
    /// `blocking` = the I48 `cont.resume.block` variant: on a still-parked fiber, idle the resumer's
    /// task on the fiber's event instead of returning `FIBER_PARKED` (still advisory — the guest keeps
    /// its loop; `FIBER_PARKED` remains a legal transient on the value-recheck path).
    ContResume {
        k: u32,
        arg: u32,
        dst: u32,
        blocking: bool,
    },
    /// §12 fiber suspend (`suspend`): hand `value` back to the resumer (status SUSPENDED) and park
    /// this fiber; `dst` receives the next resume's `arg`. Driver-driven.
    Suspend {
        value: u32,
        dst: u32,
    },
    /// `<setjmp.h>` `setjmp`: checkpoint this activation's resume point (the op after `setjmp`) keyed
    /// by the guest `jmp_buf` address in `buf`; `dst` receives `i32` 0 (or the long-jump value on
    /// re-entry). Intra-vCPU — handled inline, no scheduler escape.
    SetJmp {
        buf: u32,
        dst: u32,
    },
    /// `<setjmp.h>` `longjmp`: pop the activation stack back to the `setjmp` checkpoint named by `buf`,
    /// re-entering it with the `setjmp` result set to `val` (a `0` becomes `1`, per C). Noreturn.
    LongJmp {
        buf: u32,
        val: u32,
    },
    /// §12 `thread.spawn`: spawn a vCPU running `func` (a direct func index) with `(sp, arg)`; its
    /// handle lands at `dst`. Scheduler-driven.
    ThreadSpawn {
        func: u32,
        sp: u32,
        arg: u32,
        dst: u32,
    },
    /// §12 `thread.join`: park until child `handle` finishes; its result (or trap) lands at `dst`.
    ThreadJoin {
        handle: u32,
        dst: u32,
    },
    /// §14 `Instantiator.instantiate(entry, off, size_log2, quota)` (op 0): spawn a **confined
    /// executor child** running `entry` over `[off, off+2^size_log2)` of the holder's range, with an
    /// attenuated `Instantiator`+`AddressSpace` powerbox over its own window; its handle (or `EINVAL`)
    /// lands at `dst`. `handle` is the Instantiator cap (authority). Scheduler-driven (joinable).
    Instantiate {
        handle: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        quota: u32,
        dst: u32,
        /// §14 `instantiate_named` (op 11, PROCESS.md S2): the `(grants_ptr, grants_n)` register pair
        /// for the child's by-name grant list (op 0 is `None`). Same-module counterpart of op 13 — the
        /// child runs the holder's *own* program at `entry`, but its powerbox additionally carries the
        /// re-granted `grants_n × {name_off, name_len, handle, flags}` caps read from the parent window
        /// (via the shared `Host::spawn_named_child`), so a spawned stage resolves an inherited region
        /// (a ring end) or `stdout` by name — the concurrent-pipeline spawn.
        grants: Option<(u32, u32)>,
    },
    /// §14 `Instantiator.instantiate_module(module, entry, off, size_log2, quota)` (op 5): like
    /// [`Op::Instantiate`], but the child runs a host-granted **separate** `Module` (`module` is its
    /// handle, crossing as the first i64 arg) rather than the holder's own program — the §14
    /// "plugin-in-plugin" story. The driver resolves + compiles the module, materializes its data into
    /// the carve, and runs it as a confined executor child. `handle` is the Instantiator cap.
    InstantiateModule {
        handle: u32,
        module: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        quota: u32,
        dst: u32,
        /// §14 `instantiate_module_named` (op 13): the `(grants_ptr, grants_n)` register pair for the
        /// child's by-name grant list (op 5 is `None`). The driver reads the `grants_n × {name_off,
        /// name_len, handle, flags}` records from the parent window and re-grants each into the child
        /// powerbox (via the shared `Host::spawn_named_child`), so a spawned command resolves an
        /// inherited `stdout` by name — the shell "exec" primitive.
        grants: Option<(u32, u32)>,
    },
    /// CONSOLIDATION.md §3d — `instantiate_rec(record_ptr)` (op 17): the config-record spawn.
    /// The 56-byte record is **runtime data** (entry, carve, module, budget, grants — see the
    /// tree-walker's op-17 arm for the layout), so unlike the scalar spawns above the fields are
    /// read from the vCPU's confined window at **exec** time; the op then folds onto the same
    /// [`Outcome::Instantiate`] / [`Outcome::InstantiateModule`] the drivers already service.
    InstantiateRec {
        handle: u32,
        rec: u32,
        dst: u32,
    },
    /// §14 `Instantiator.join(child)` (op 1): park until executor child `child` finishes; its result
    /// (or trap) lands at `dst`. `handle` is the Instantiator cap (authority). The join itself reuses
    /// the §12 thread machinery — children share one handle namespace (`threads`) with `thread.spawn`.
    InstJoin {
        handle: u32,
        child: u32,
        dst: u32,
    },
    /// §12 `memory.wait`: futex wait (`ty`-wide) on `addr` while it equals `expected`, up to
    /// `timeout` ns; the status (0/1/2) lands at `dst`. Scheduler-driven.
    MemoryWait {
        ty: IntTy,
        addr: u32,
        expected: u32,
        timeout: u32,
        dst: u32,
    },
    /// §12 `memory.notify`: wake up to `count` waiters on `addr`; the woken count lands at `dst`.
    MemoryNotify {
        addr: u32,
        count: u32,
        dst: u32,
    },
    /// §22 `Jit.install(code)` (op 3): compile the unit named by code-handle `code` to bytecode and
    /// install it into the domain's dispatch table; the slot (or `-ENOSPC`) lands at `dst`. `handle`
    /// is the `Jit` domain cap (authority).
    JitInstall {
        handle: u32,
        code: u32,
        dst: u32,
    },
    /// §22 `Jit.uninstall(slot)` (op 4): clear an installed table slot; `0`/`EINVAL` lands at `dst`.
    JitUninstall {
        handle: u32,
        slot: u32,
        dst: u32,
    },
    /// §22 `Jit.invoke(code, args…)` (op 1): run the unit named by `code` synchronously over the
    /// shared window/powerbox; its results land at `dst…`. `params`/`results` are the unit entry's
    /// expected signature (the `cap.call` sig minus the leading code-handle param), used to marshal
    /// args/results through the i64-slot ABI.
    JitInvoke {
        handle: u32,
        code: u32,
        args: Box<[u32]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §GC `gc.roots(heap_lo, heap_hi, mask, buf, cap)`: conservative root enumeration. Escapes to
    /// the driver, which scans every live activation of the vCPU's continuation (the active window,
    /// its call stack, its resume-chain ancestors, parked fibers, and coroutines) for words that —
    /// masked — land in `[lo, hi)`, writes the first `cap` (ascending, deduplicated) to guest memory
    /// at `buf`, and writes the total found to `dst`. Sound (a superset of the genuine roots), not
    /// bit-identical to the tree-walker — the backends over-approximate differently (GC.md §3.2).
    GcRoots {
        lo: u32,
        hi: u32,
        mask: u32,
        buf: u32,
        cap: u32,
        dst: u32,
    },
    Ret {
        srcs: Box<[u32]>,
    },
    /// `return_call`: a direct tail call — reuse the current activation window (no stack growth),
    /// staying in the caller's module; on return the callee returns to *this* activation's caller.
    TailCall {
        callee: u32,
        args: Box<[u32]>,
    },
    /// `return_call_indirect`: an indirect tail call — resolve through the runtime dispatch table
    /// (possibly cross-module), then reuse the current window like [`Op::TailCall`].
    TailCallIndirect {
        idx: u32,
        args: Box<[u32]>,
        want_params: Box<[ValType]>,
        want_results: Box<[ValType]>,
    },
    Unreachable,
    /// Long-tail value/store ops (SIMD, `v128` load/store, fences) delegated to the reference
    /// [`super::eval_inst`] — same semantics, no duplication. The original instruction keeps its
    /// **block-local** operand indices, so it's run against the sub-window `regs[base + block_base
    /// ..]`; `dst` is the frame-relative result slot (unused when `eval_inst` yields no value).
    Eval {
        inst: Box<Inst>,
        block_base: u32,
        dst: u32,
    },
    /// §12.8 4A.5 durable-runtime-internal: push the active context's shadow-SP word address (the
    /// `Vm`'s `durable_region_base`). The reference `eval_inst` can't service it (it needs the running
    /// context), so it gets a dedicated op like `vcpu.tls` would.
    DurableShadowBase {
        dst: u32,
    },
    /// §12 per-vCPU **thread-local register** read (`vcpu.tls.get`): push this `Vm`'s `tls` word to
    /// `dst`. The reference `eval_inst` traps `Malformed` on it (no vCPU context), so — like
    /// [`Op::DurableShadowBase`] — it gets a dedicated op rather than the `Eval` fallback. Seeded to
    /// the dense vCPU id (root = 0) at `Vm` construction; a spawned thread's `Vm` is re-seeded to its
    /// id (see `drive`'s `Spawn` arm). See the tree-walker's `Inst::VcpuTlsGet`.
    VcpuTlsGet {
        dst: u32,
    },
    /// §12 per-vCPU **thread-local register** write (`vcpu.tls.set`): set this `Vm`'s `tls` word to
    /// `val`. No result (like `store`).
    VcpuTlsSet {
        val: u32,
    },
}

/// Marks a [`Program::src`] entry as a **terminator** op's location (OR-ed into the `inst` field).
/// Two readers need terminators distinguished from instructions: [`Vm::cur_ir_pc`] (debug stepping)
/// skips them, while [`vm_trap_bt`] (trap backtrace) *reports* them — a trap at a terminator
/// (`unreachable`, `return_call_indirect`) is real and the tree-walker names it. The flag is the high
/// bit, never set by a real block/inst count, so masking it off recovers the stored index.
const SRC_TERM: u32 = 1 << 31;

struct Program {
    ops: Vec<Op>,
    nslots: u32,
    /// Debug reverse map (Slice 1c-3): the source `(block, inst)` of each op. An instruction op maps
    /// to its `(block, inst)`; a **terminator** op maps to `(block, insts.len() | `[`SRC_TERM`]`)` —
    /// the `insts.len()` is the `inst` the tree-walker's `Vec<Frame>` carries for a terminator (it sits
    /// one past the block's last instruction). The tree-walker's debug seam (`run_inner`'s `before_op`)
    /// stops only at **instructions**, never terminators, so [`Vm::cur_ir_pc`] reports `None` for a
    /// [`SRC_TERM`] entry — keeping the engine's step/breakpoint location trace identical to the
    /// tree-walker's [`crate::IrPc`] sequence op-for-op — while [`vm_trap_bt`] still resolves it for a
    /// trap-time backtrace.
    src: Box<[Option<(u32, u32)>]>,
}

/// A whole compiled module: one [`Program`] per function plus each function's result types (for
/// reconstructing typed `Value`s at the entry boundary).
pub struct Compiled {
    progs: Vec<Program>,
    result_types: Vec<Vec<ValType>>,
    /// Per-function `(params, results)` for `call_indirect` type-checking — the natural module-0
    /// function table indexes these directly (slot `i` ⇒ func `i`).
    sigs: Vec<(Vec<ValType>, Vec<ValType>)>,
    /// `len - 1` of the natural table (`next_power_of_two(n_funcs)`), used to mask a `ref.func`/fiber
    /// funcref to a module-local slot (the fiber/coroutine dispatch is module-0-natural).
    table_mask: usize,
}

impl Compiled {
    /// Total compiled **bytecode op count** across all functions — the structural-size measure of this
    /// threaded register-VM program (the analogue of the JIT's emitted code bytes / the IR's
    /// instruction count). The engine is a `Vec<Op>` per function, not a serialized byte stream, so op
    /// count, not a byte length, is the meaningful size.
    pub fn op_count(&self) -> usize {
        self.progs.iter().map(|p| p.ops.len()).sum()
    }
}

/// THREADS.md 4c-domain — a domain's `call_indirect` dispatch table, **shareable + installable across
/// parallel vCPUs** (mirrors the tree-walker's [`crate::DomainTable`]). Each slot packs `(module, func)`
/// (`module<<32 | func`, [`super::pack_slot`]); `module == TABLE_EMPTY` is trapping padding. Dispatch is
/// one `Acquire` load; `install` does a `Release` store, so a vCPU that observes a filled slot also
/// observes the unit pushed into the [`ModuleSource`] before it (the install serializes under the
/// source lock). Built once per domain (root / §14 child / coroutine); only the root's is installed into.
struct SharedSlots {
    slots: Box<[std::sync::atomic::AtomicU64]>,
}

impl SharedSlots {
    /// `2^table_log2` (at least `next_power_of_two(n_funcs)`) slots: the first `n_funcs` map to
    /// `(module, i)` (module 0 for the primary's natural table; a `k≥1` for a §14 separate-module
    /// child), the rest are trapping padding (fillable by [`Domain::install`]).
    fn new(n_funcs: usize, table_log2: u8, module: u32) -> SharedSlots {
        let len = (1usize << table_log2)
            .max(n_funcs.next_power_of_two())
            .max(1);
        let slots = (0..len)
            .map(|i| {
                std::sync::atomic::AtomicU64::new(if i < n_funcs {
                    super::pack_slot(module, i as u32)
                } else {
                    super::pack_slot(super::TABLE_EMPTY, 0)
                })
            })
            .collect();
        SharedSlots { slots }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    /// Dispatch-path read: one `Acquire` load, paired with [`Domain::install`]'s `Release` store.
    #[inline]
    fn slot(&self, i: usize) -> super::TableSlot {
        super::unpack_slot(self.slots[i].load(std::sync::atomic::Ordering::Acquire))
    }
}

/// THREADS.md 4c-domain — a domain's compiled modules, **shared (`Arc`) and append-only** so installed
/// §22 units / §14 separate-module children are visible to every parallel vCPU without invalidating
/// references the way a growing `Vec<Compiled>` would: the modules live behind `Arc<Compiled>` (stable
/// address) inside a `Mutex<Vec<_>>` touched only on install or a reader's local-cache miss. `mods[0]`
/// is the primary; `k≥1` is an installed unit. A §14 child / coroutine shares the root's `ModuleSource`
/// (so its table's module indices resolve) but carries its own [`SharedSlots`].
struct ModuleSource {
    mods: std::sync::Mutex<Vec<std::sync::Arc<Compiled>>>,
}

impl ModuleSource {
    fn new(primary: Compiled) -> ModuleSource {
        ModuleSource {
            mods: std::sync::Mutex::new(vec![std::sync::Arc::new(primary)]),
        }
    }

    /// A fresh clone of the module `Arc`s — a vCPU's lock-free local cache (cheap refcount bumps),
    /// refreshed on a miss. The lock acquire pairs with `install`'s push, so the snapshot sees it.
    fn snapshot(&self) -> Vec<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned().clone()
    }

    /// The primary program (module 0).
    fn primary(&self) -> std::sync::Arc<Compiled> {
        std::sync::Arc::clone(&self.mods.lock_unpoisoned()[0])
    }

    /// Module `i` (`0` = primary, `k≥1` = an installed unit), or `None` if out of range.
    fn get(&self, i: usize) -> Option<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned().get(i).cloned()
    }

    /// Append a module (a §14 `instantiate_module` child's program) and return its index. (§22
    /// `Jit.install` instead goes through [`Domain::install`], which also fills a dispatch slot.)
    fn push(&self, unit: Compiled) -> usize {
        let mut mods = self.mods.lock_unpoisoned();
        mods.push(std::sync::Arc::new(unit));
        mods.len() - 1
    }

    /// The **non-primary** units (`mods[1..]`) — a time-travel checkpoint captures these (cheap `Arc`
    /// refcount bumps) so a reverse-`seek` restore can re-push them and a separate-module coroutine/child
    /// frame's `module` index resolves as it did at capture. Paired with [`reset_extra`].
    fn extra_units(&self) -> Vec<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned()[1..].to_vec()
    }

    /// Reset the pushed units to exactly `units` (keeping the primary at index 0) — the restore inverse
    /// of [`extra_units`]. Idempotent, so restoring twice into the same run is safe.
    fn reset_extra(&self, units: &[std::sync::Arc<Compiled>]) {
        let mut mods = self.mods.lock_unpoisoned();
        mods.truncate(1);
        mods.extend(units.iter().cloned());
    }
}

/// Build a §14 child / coroutine's natural dispatch table over its `module` in the shared source.
fn build_table_for(n_funcs: usize, table_log2: u8, module: u32) -> SharedSlots {
    SharedSlots::new(n_funcs, table_log2, module)
}

/// Build the primary's natural module-0 dispatch table.
fn build_table(n_funcs: usize, table_log2: u8) -> SharedSlots {
    SharedSlots::new(n_funcs, table_log2, 0)
}

/// A running domain (THREADS.md 4c-domain): its shared [`ModuleSource`] (`mods[0]` = primary, `k≥1` =
/// installed §22 units / §14 child modules) plus its own [`SharedSlots`] `call_indirect` dispatch table.
/// Both parts are interior-mutable + thread-safe, so a **parallel** driver can share `&Domain` across
/// vCPU threads and still `install`; the cooperative path is single-threaded (uncontended atomics/lock,
/// so dispatch order — hence determinism — is unchanged). A §14 child / coroutine shares the root's
/// `source` (its table's module indices resolve there) but carries its own `table`.
struct Domain {
    source: std::sync::Arc<ModuleSource>,
    table: SharedSlots,
}

impl Domain {
    fn new(primary: Compiled, table_log2: u8) -> Domain {
        let table = SharedSlots::new(primary.progs.len(), table_log2, 0);
        Domain {
            source: std::sync::Arc::new(ModuleSource::new(primary)),
            table,
        }
    }

    /// A §14 confined-child domain over a (cloned `Arc`) **shared** `source` with its own dispatch
    /// `table`. Sharing the source keeps the parent's module archive reachable by index (so an
    /// `instantiate_module` child's pushed program resolves); the fresh `table` is the confinement —
    /// it carries only the child's own natural entries, never the parent's installed §22 unit slots
    /// (matching the tree-walker's `DomainTable::new(&cfuncs, 0)`).
    fn child(source: std::sync::Arc<ModuleSource>, table: SharedSlots) -> Domain {
        Domain { source, table }
    }

    /// `Jit.install`: append `unit` to the shared source and fill the first padding slot with
    /// `(module, 0)`, returning the slot — or `None` if the table is full (`-ENOSPC`; the unit is not
    /// appended). `&self` (interior-mutable) so a shared `&Domain` can install. See [`jit_install_into`].
    fn install(&self, unit: Compiled) -> Option<usize> {
        jit_install_into(&self.source, &self.table, unit)
    }

    /// `Jit.uninstall`: clear a filled padding slot (`≥ n_real`) back to trapping. See
    /// [`jit_uninstall_from`].
    fn uninstall(&self, slot: usize, n_real: usize) -> bool {
        jit_uninstall_from(&self.source, &self.table, slot, n_real)
    }
}

/// `Jit.install` over a raw `(source, table)` pair — the shared body of [`Domain::install`] and the
/// debug engines' `dbg_jit_install` (`DebugRun`/`ScheduledDebugRun` hold `source`/`table` as separate
/// fields, not a wrapped [`Domain`]). Append `unit` to the shared source and fill the first padding
/// slot with `(module, 0)`, returning the slot — or `None` if the table is full (`-ENOSPC`; the unit
/// is not appended). The whole op serializes under the source lock, and the slot store is `Release`,
/// so a reader that observes the slot also observes the pushed unit.
fn jit_install_into(source: &ModuleSource, table: &SharedSlots, unit: Compiled) -> Option<usize> {
    use std::sync::atomic::Ordering;
    let mut mods = source.mods.lock_unpoisoned();
    let slot = table
        .slots
        .iter()
        .position(|s| (s.load(Ordering::Relaxed) >> 32) as u32 == super::TABLE_EMPTY)?;
    mods.push(std::sync::Arc::new(unit));
    let module = (mods.len() - 1) as u32;
    table.slots[slot].store(super::pack_slot(module, 0), Ordering::Release);
    Some(slot)
}

/// `Jit.uninstall` over a raw `(source, table)` pair — the shared body of [`Domain::uninstall`] and
/// the debug engines' `dbg_jit_uninstall`. Clear a filled padding slot (`≥ n_real`) back to trapping,
/// returning success. A real-function slot (`< n_real`), out-of-range, or already-empty slot is
/// rejected. The unit stays in `source` (append-only); only the slot is reclaimed. Serialized under
/// the source lock.
fn jit_uninstall_from(
    source: &ModuleSource,
    table: &SharedSlots,
    slot: usize,
    n_real: usize,
) -> bool {
    use std::sync::atomic::Ordering;
    let _g = source.mods.lock_unpoisoned();
    if slot >= n_real
        && slot < table.slots.len()
        && (table.slots[slot].load(Ordering::Relaxed) >> 32) as u32 != super::TABLE_EMPTY
    {
        table.slots[slot].store(super::pack_slot(super::TABLE_EMPTY, 0), Ordering::Release);
        true
    } else {
        false
    }
}

/// The concurrency/park seams a module's instructions touch — one linear scan feeding both the
/// [`compile_module`] combination vetoes and the cross-backend serve qualification
/// ([`serve_qualifies`]).
#[derive(Default)]
struct Seams {
    has_coro: bool,
    has_fiber: bool,
    has_thread: bool,
    has_instantiate: bool,
    has_gc: bool,
    has_svc: bool,
    has_park_seam: bool,
    /// FORK.md §9 — `clone_caller` (self-op 11) present: the fork-returns-twice servicer primitive.
    /// A distinct seam because the **bytecode** engine now services it natively ([`bytecode_serves_fork`]),
    /// while the Cranelift routing still folds it (`svc_park_veto` keeps it). (`reap`, self-op 12, is
    /// **not** here yet — it stays a `has_park_seam` so fork+wait still folds; that is the next slice.)
    has_fork: bool,
}

impl Seams {
    /// The **serve-qualification veto**: a service point (`svc.poll` / `svc.wait`) coexisting with
    /// any seam that could park or unwind a handler mid-dispatch. This is the single definition of
    /// that disjunction — consulted by both the bytecode compile gate ([`compile_module`]) and the
    /// exported [`serve_qualifies`] that svm-run's JIT routing folds on — so the two backends can
    /// never drift over which modules serve natively vs. decline to the tree-walk oracle. Adding a
    /// new park-capable seam means extending this one list. (INVARIANTS.md §9: one veto predicate,
    /// one definition.)
    ///
    /// **Fork rides this veto for Cranelift** (`has_fork` is in the disjunction), so svm-run's JIT
    /// routing still folds a forking module to the oracle — the Cranelift fork slice is unbuilt
    /// (FORK.md §9.1). The **bytecode** engine, in contrast, services `clone_caller`/`reap` natively;
    /// its compile gate takes the [`bytecode_serves_fork`] escape past this veto (a per-backend split,
    /// the one the two-predicate structure exists to allow — the bytecode gate and `serve_qualifies`
    /// legitimately diverge on fork until Cranelift catches up).
    fn svc_park_veto(&self) -> bool {
        self.has_svc
            && (self.has_park_seam
                || self.has_fiber
                || self.has_thread
                || self.has_coro
                || self.has_instantiate
                || self.has_gc
                || self.has_fork)
    }

    /// FORK.md §9.2 — the **bytecode fork-serving escape**: a serving module the bytecode engine can
    /// run `clone_caller` in natively even though [`svc_park_veto`] folds it (for Cranelift). The
    /// bounded shape: it serves (`has_svc`) and forks (`has_fork`), the manager may spawn children
    /// (`has_instantiate` — the fork topology needs it), and **no other** seam that could park a
    /// *handler* mid-dispatch is present. `clone_caller` itself never parks the handler (it reshapes
    /// the parked caller and returns), so the serve rewind linkage stays intact; the manager's
    /// instantiate/join park ordinary tasks, not handlers. Deliberately narrow (fork-shaped modules
    /// only) to bound the blast radius vs. the general serve+spawn case, which stays folded. A fork
    /// handler that *also* parked (e.g. joined) is out of this shape and is not admitted here.
    fn bytecode_serves_fork(&self) -> bool {
        self.has_svc
            && self.has_fork
            && !self.has_park_seam
            && !self.has_fiber
            && !self.has_thread
            && !self.has_coro
            && !self.has_gc
    }
}

fn scan_seams(funcs: &[Func]) -> Seams {
    let mut s = Seams::default();
    for f in funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    // ops 0/1 = instantiate/join, op 5 = instantiate_module, op 13 =
                    // instantiate_module_named, op 17 = instantiate_rec (all executor children,
                    // scheduler-driven — the grant-carrying spawns re-grant caps but spawn the same
                    // kind of confined task); everything else on INSTANTIATOR is the legacy coroutine
                    // residue. Classifying the named spawns as `has_instantiate` (not `has_coro`) is
                    // load-bearing: a concurrent pipeline mixes them with `memory.wait`/`notify`
                    // (`has_thread`), and the `has_coro && has_thread` veto would otherwise fall the
                    // whole module back to the tree-walker.
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR,
                        op: 0 | 1 | 5 | 13 | 14 | 17,
                        ..
                    } => s.has_instantiate = true,
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR,
                        ..
                    } => s.has_coro = true,
                    // I38's **timed** `svc.wait` (op 10 with the optional timeout arg) needs
                    // the scheduler's deadline machinery — oracle-only; veto like a park seam
                    // so both fast backends decline the module.
                    Inst::CapCall {
                        type_id: svm_ir::CAP_SELF_TYPE_ID,
                        op: 10,
                        args,
                        ..
                    } if !args.is_empty() => {
                        s.has_svc = true;
                        s.has_park_seam = true;
                    }
                    // §3.6 service points (I36 slice 1): svc.poll/svc.wait sites — natively
                    // servable only when nothing in the module could park a handler (below).
                    Inst::CapCall {
                        type_id: svm_ir::CAP_SELF_TYPE_ID,
                        op: 9 | 10,
                        ..
                    } => s.has_svc = true,
                    // FORK.md §9 — `clone_caller` (11) / `reap` (12): the fork servicer primitives.
                    // The bytecode engine services both natively (the [`bytecode_serves_fork`] escape
                    // admits the fork topology; the `VcpuStop::CloneCaller`/`Reap` driver arms build
                    // the twin and reap it), so they are the `has_fork` seam — folded for Cranelift
                    // (`svc_park_veto` keeps `has_fork`) but run natively on bytecode.
                    Inst::CapCall {
                        type_id: svm_ir::CAP_SELF_TYPE_ID,
                        op: 11 | 12,
                        ..
                    } => s.has_fork = true,
                    // A blocking stream `read` (type 0 op 0) can stdin-park, and an import call
                    // can be *bound* to one at spawn — either inside a handler would need the
                    // tree-walker's FIBER_PARKED (completed-but-not-replied) machinery.
                    Inst::CapCall {
                        type_id: super::cap_id::STREAM,
                        op: 0,
                        ..
                    }
                    | Inst::CapCall {
                        type_id: svm_ir::CAP_IMPORT_TYPE_ID,
                        ..
                    }
                    | Inst::CallImport { .. }
                    | Inst::SetJmp { .. }
                    | Inst::LongJmp { .. } => s.has_park_seam = true,
                    Inst::ContNew { .. }
                    | Inst::ContResume { .. } // I48: `block` flag is advisory here
                    | Inst::Suspend { .. } => s.has_fiber = true,
                    Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. } => s.has_thread = true,
                    Inst::GcRoots { .. } => s.has_gc = true,
                    _ => {}
                }
            }
        }
    }
    s
}

/// §3.6 (I36): the **serve qualification** — `funcs` contain a service point (`svc.poll` /
/// `svc.wait`) and no seam that could park or unwind a handler mid-dispatch, so a fast backend
/// may run the serve loop natively (every handler runs to completion or traps; the tree-walk
/// oracle's fiber-park machinery is never needed). The veto is module-wide, so it covers
/// handlers' transitive callees for free. This is the same predicate [`compile_module`]'s veto
/// applies — exported so svm-run's JIT routing folds exactly the modules this engine declines
/// (one definition, no drift). A module with no service point returns `false` (it has nothing
/// to serve natively; the caller decides what that means).
pub fn serve_qualifies(funcs: &[Func]) -> bool {
    let s = scan_seams(funcs);
    s.has_svc && !s.svc_park_veto()
}

/// Lower every function (fast path — superinstruction-**fused**, Slice 5a), or `None` if any uses an
/// op outside this slice's subset. This is what every production/runtime path calls.
/// CONSOLIDATION.md §3d — the **module-level** admission for the record spawn (op 17): a module
/// that could build a *pager* record (it has impl exports) declines to the tree-walk oracle, which
/// owns demand paging; with no impl exports every pager record `CapFault`s identically on every
/// tier, so the exec arm's fail-closed pager check is exact. This mirrors svm-run's
/// `module_demand_spawns` fold on the Cranelift tier — one predicate per tier boundary, consulted
/// by every `&Module` compile entry (INVARIANTS.md §9).
fn compile_module_for(m: &Module) -> Option<Compiled> {
    let uses_rec = m.funcs.iter().flat_map(|f| f.blocks.iter()).any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::CapCall {
                    type_id: super::cap_id::INSTANTIATOR,
                    op: 17,
                    ..
                }
            )
        })
    });
    // §3d pager guard: an op-17 record-spawn module with impl exports *could* build a pager record
    // (which behaves differently), so it folds to the oracle — **except** a fork-shaped module
    // (FORK.md §9.2): its op-17 records are ordinary executor spawns (the manager spawning the
    // server/guest with by-name grants — the only in-module spawn-with-grants the bytecode tier
    // drives), and its impl export is the fork server. `bytecode_serves_fork` bounds this to the
    // fork shape; other serving record-spawn modules still fold.
    if uses_rec && !m.impl_exports.is_empty() && !scan_seams(&m.funcs).bytecode_serves_fork() {
        return None;
    }
    compile_module(&m.funcs)
}

/// §3d — validate + **drain** a spawn record's `Budget` at a driver's commit site, returning the
/// child's funded fuel. `Ok(None)` = refuse the spawn `-EINVAL` with the budget **intact** (mem
/// quota short, or the compiled-tier narrowed gap: a bounded spawn ceiling / bounded-zero fuel,
/// which this tier — like the Cranelift thunk — cannot represent; flip those when child vCPU
/// quotas / zero-fuel children land here). `Err(CapFault)` = the handle vanished since the exec
/// arm's peek (a shared-powerbox race). The fund rule is the tree-walker's: bounded fuel is
/// `min(budget, parent_remaining)`, unbounded inherits the parent's remaining.
fn take_spawn_budget(
    host: &mut Host,
    budget: i32,
    child_size: u64,
    parent_fuel: u64,
) -> Result<Option<u64>, Trap> {
    // Peek + mem-quota gate is the shared `Host::budget_for_spawn` (#911): `None` = dangling handle
    // (CapFault), `Some(Err(()))` = bounded mem quota short (refuse `-EINVAL`, budget intact).
    let (fuel, spawn) = match host.budget_for_spawn(budget, child_size, false) {
        None => return Err(Trap::CapFault),
        Some(Err(())) => return Ok(None), // mem quota short — refuse, budget intact
        Some(Ok(v)) => v,
    };
    if spawn >= 0 || fuel == 0 {
        return Ok(None); // narrowed gap (see doc) — refuse, budget intact
    }
    // Commit: drain to zero. `take_budget` returns the pre-drain state, so the fuel it reports
    // equals the `fuel` peeked above — fund from that. Bounded fuel is `min(budget, parent)`,
    // unbounded inherits the parent's remaining.
    host.take_budget(budget).ok_or(Trap::CapFault)?;
    Ok(Some(if fuel >= 0 {
        (fuel as u64).min(parent_fuel)
    } else {
        parent_fuel
    }))
}

/// The §14 carve **geometry** check (D19), the single definition every spawn/instantiate driver and
/// tree-walk arm shares: a child window of `1 << size_log2` bytes at offset `off` fits inside the
/// parent's `[0, isize)` iff the size is a valid power of two (`size_log2` in `0..64`), `off` is
/// size-aligned, and the whole span lies within `isize`. Overflow-free (an out-of-range `size_log2`
/// or a `off + size` past `u64` yields `false`, not a shift/overflow). A future bound tweak lives
/// here, not in ten pasted copies (#911).
///
/// #964: the carve may also not dip into the holder window's reserved NULL region — `ibase + off`
/// (the carve's window-relative base; `ibase` is the holder's own base) must clear `null_guard`
/// (`0` = unguarded, trivially true). The host seeds/copies a carve outside the guarded call, and
/// the reserved region is permanent by design, so a below-guard carve is refused, not admitted.
pub(crate) fn carve_fits(
    off: u64,
    size_log2: i64,
    isize: u64,
    ibase: u64,
    null_guard: u64,
) -> bool {
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    child_size != 0
        && child_size <= isize
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= isize)
        && ibase.checked_add(off).is_some_and(|b| b >= null_guard)
}

/// A §14 child module's **entry signature** must be `(i64) -> (i64)` (an instantiator handle) or
/// `(i64, i64) -> (i64)` (also an address-space handle, so the child manages its own pages). The
/// single definition the drivers and tree-walk arms share (#911).
pub(crate) fn child_entry_ok(params: &[ValType], results: &[ValType]) -> bool {
    results == [ValType::I64]
        && (params == [ValType::I64] || params == [ValType::I64, ValType::I64])
}

pub fn compile_module(funcs: &[Func]) -> Option<Compiled> {
    compile_module_with(funcs, true)
}

/// Unfused lowering — one op per source instruction, so the step/location trace stays
/// tree-walker-identical. The debug/trace entries (`ir_trace`, `ir_window_trace`, `ir_value_trace`,
/// `debug_advance_fiber`, `dbg_pick_runnable`) use this; results and traps are identical to the fused
/// form (fusion only merges a pure compare into its sole-consumer branch).
pub fn compile_module_unfused(funcs: &[Func]) -> Option<Compiled> {
    compile_module_with(funcs, false)
}

/// Lower every function, or `None` if any uses an op outside this slice's subset.
fn compile_module_with(funcs: &[Func], fuse: bool) -> Option<Compiled> {
    // Coroutines (§14, `spawn_coroutine`/`resume`/`yield`) are driven **inline** as single-vCPU
    // children with a Yielder-only powerbox. A coroutine module that *also* uses fibers or threads
    // would need the child to participate in those seams (a coroutine child can use `cont.*`/`thread.*`
    // in the tree-walker), which the inline coroutine driver here doesn't service — so reject the
    // combination (→ tree-walker fallback). §14 **executor children** (`instantiate`/`join`, ops 0/1)
    // are different: they run on the scheduler like threads, not inline — so they classify as
    // scheduler-driven, not as coroutines. The one combination they can't yet service is `cont.*`
    // fibers (a confined child would share the run-shared fiber registry — a divergence), so reject
    // instantiate+fiber. Plain coroutine / fiber / thread / instantiate modules are each fine, as are
    // instantiate+thread and instantiate+coroutine.
    let s = scan_seams(funcs);
    // `gc.roots` (§GC) is per-vCPU **conservative root enumeration**: on this engine it scans the
    // calling vCPU's continuation (`vt.active` + `vt.chain` + `vt.coroutines`) **plus the run-shared
    // fiber registry** (`fibers`, scanned in [`step_vcpu`]'s `Outcome::GcRoots` arm) — the exact
    // scope the tree-walker's op documents ("the caller's own live frames, the parked root, and every
    // registry fiber's frames", `crates/svm/tests/gc_roots.rs`). Neither engine scans a *sibling
    // thread's* own frames, and neither has to: a guest GC that threads coordinates a stop-the-world
    // quiesce and has each vCPU enumerate its own roots (the reference barrier is
    // `crates/svm/tests/gc_quiesce.rs`); JACL's roots live in the migratable fibers the shared
    // registry covers. So `gc.roots` + `thread.*` is **not** vetoed — the criterion for this op is
    // soundness (`tw ⊆ bc`, GC.md §3.2), which holds because the bytecode scope is a superset of the
    // tree-walker's. (`gc.roots` + fibers / coroutines was always fine — those continuations are
    // scanned.)
    //
    // §3.6 (I36 slice 1): a **serving** module is admitted natively only when no handler could
    // park or unwind mid-dispatch ([`serve_qualifies`]) — any park-capable seam anywhere in the
    // module (futex waits / threads, fibers, coroutines, nested instantiate, setjmp/longjmp — a
    // `longjmp` out of a handler would unwind past the serve linkage — blocking stream reads,
    // spawn-bound imports, gc.roots) falls the whole module back to the tree-walk oracle, whose
    // serve arm has the fiber-park machinery (slice 5b).
    // FORK.md §9.2 — the bytecode fork-serving escape: a fork-shaped module (`bytecode_serves_fork`)
    // is admitted natively even though `svc_park_veto` folds it for Cranelift (the per-backend split).
    if (s.has_coro && (s.has_fiber || s.has_thread))
        || (s.has_instantiate && s.has_fiber)
        || (s.svc_park_veto() && !s.bytecode_serves_fork())
    {
        return None;
    }

    let arities: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    let mut progs = Vec::with_capacity(funcs.len());
    for f in funcs {
        progs.push(compile_func(f, &arities, fuse)?);
    }
    let table_mask = funcs.len().next_power_of_two().max(1) - 1;
    Some(Compiled {
        progs,
        result_types: funcs.iter().map(|f| f.results.clone()).collect(),
        sigs: funcs
            .iter()
            .map(|f| (f.params.clone(), f.results.clone()))
            .collect(),
        table_mask,
    })
}

fn compile_func(f: &Func, arities: &[usize], fuse: bool) -> Option<Program> {
    // Global slot per value: each block's params then its value-producing insts, in order.
    let mut base = Vec::with_capacity(f.blocks.len());
    let mut nslots = 0u32;
    for b in &f.blocks {
        base.push(nslots);
        nslots += b.params.len() as u32;
        for inst in &b.insts {
            nslots += inst.result_count(arities) as u32;
        }
    }
    let mut block_pc = vec![0u32; f.blocks.len()];
    let mut ops: Vec<Op> = Vec::new();
    // Debug reverse map (Slice 1c-3), built **incrementally** alongside `ops` (was a positional
    // post-pass) so fusion — which drops an op — keeps `src` and `ops` in lockstep: each op push is
    // paired with exactly one `src` push. Instruction ops map to their `(block, inst)`; the
    // terminator op maps to `(block, insts.len() | SRC_TERM)` (flagged so `cur_ir_pc` skips it while
    // `vm_trap_bt` can name a terminator-trap site). A fused `BrIfCmp` takes the terminator location
    // (it can never trap), and the fused-away `IntCmp`'s entry is dropped — the fused program is
    // never single-stepped (debug/trace compile unfused), so no source location is lost there.
    let mut src: Vec<Option<(u32, u32)>> = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        block_pc[bi] = ops.len() as u32;
        let g = |local: u32| base[bi] + local; // operand: block-local index -> frame slot
        let mut local = b.params.len() as u32;
        for (i, inst) in b.insts.iter().enumerate() {
            let dst = base[bi] + local;
            local += inst.result_count(arities) as u32;
            ops.push(compile_inst(inst, dst, base[bi], &g)?);
            src.push(Some((bi as u32, i as u32)));
        }
        // Terminator -> edge copies (block-local src in this block -> first slots of target) + jump.
        let edge = |bidx: usize, args: &[u32]| -> Edge {
            // Slice 5b: drop **identity** self-copies (`src == dst`) at compile time. A loop-invariant
            // block param threaded unchanged across a back-edge lands in the same global slot it came
            // from, so the copy is a no-op — eliding it removes a real `scratch` push+write per such
            // param every iteration. Safe and semantics-transparent: an `x -> x` move changes nothing,
            // and its removal can't affect the gather/scatter of the other (aliasing) copies. Applies
            // uniformly to every terminator's edges (Br/BrIf/BrIfCmp/BrTable), fused or not.
            let pairs: Box<[(u32, u32)]> = args
                .iter()
                .enumerate()
                .map(|(i, a)| (g(*a), base[bidx] + i as u32))
                .filter(|(src, dst)| src != dst)
                .collect();
            (Copies::new(pairs), bidx as u32) // block index; patched to entry pc below
        };
        match &b.term {
            Terminator::Br { target, args } => {
                let (copies, t) = edge(*target as usize, args);
                ops.push(Op::Br { copies, target: t });
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (then_copies, tt) = edge(*then_blk as usize, then_args);
                let (else_copies, et) = edge(*else_blk as usize, else_args);
                let cond_slot = g(*cond);
                // Slice 5a: fuse a block-final `IntCmp` whose result is this branch's condition and
                // is used nowhere else into a single `BrIfCmp`. Valid because the compare is pure and
                // single-use *here*: it is the last instruction (so no later in-block reader) and the
                // cond slot is not carried to any successor (not an edge-copy source). Fused compile
                // only — `fuse == false` (debug/trace) keeps the compare as its own steppable op.
                let fused = if fuse && !b.insts.is_empty() {
                    match ops.last() {
                        Some(Op::IntCmp {
                            dst,
                            a,
                            b: cb,
                            ty,
                            op,
                        }) if *dst == cond_slot
                            && !then_copies.pairs.iter().any(|(s, _)| *s == cond_slot)
                            && !else_copies.pairs.iter().any(|(s, _)| *s == cond_slot) =>
                        {
                            Some((*a, *cb, *ty, *op))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some((a, cb, ty, op)) = fused {
                    ops.pop(); // drop the now-fused IntCmp op ...
                    src.pop(); // ... and its source-map entry (kept in lockstep)
                    ops.push(Op::BrIfCmp {
                        a,
                        b: cb,
                        ty,
                        op,
                        then_copies,
                        then_pc: tt,
                        else_copies,
                        else_pc: et,
                    });
                } else {
                    ops.push(Op::BrIf {
                        cond: cond_slot,
                        then_copies,
                        then_pc: tt,
                        else_copies,
                        else_pc: et,
                    });
                }
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let arms = targets.iter().map(|(t, a)| edge(*t as usize, a)).collect();
                let default = edge(default.0 as usize, &default.1);
                ops.push(Op::BrTable {
                    idx: g(*idx),
                    arms,
                    default,
                });
            }
            Terminator::Return(vs) => ops.push(Op::Ret {
                srcs: vs.iter().map(|v| g(*v)).collect(),
            }),
            Terminator::Unreachable => ops.push(Op::Unreachable),
            // Tail calls reuse the current activation window (no stack growth): a direct tail call
            // stays in the caller's module; an indirect one dispatches through the runtime table.
            Terminator::ReturnCall { func, args } => ops.push(Op::TailCall {
                callee: *func,
                args: args.iter().map(|a| g(*a)).collect(),
            }),
            Terminator::ReturnCallIndirect { ty, idx, args } => ops.push(Op::TailCallIndirect {
                idx: g(*idx),
                args: args.iter().map(|a| g(*a)).collect(),
                want_params: ty.params.clone().into(),
                want_results: ty.results.clone().into(),
            }),
        }
        // Exactly one terminator op was pushed above (fused `BrIfCmp` or a plain terminator); pair it
        // with the terminator source entry so `src` stays the same length as `ops`.
        src.push(Some((bi as u32, b.insts.len() as u32 | SRC_TERM)));
    }
    debug_assert_eq!(
        ops.len(),
        src.len(),
        "src map must stay in lockstep with ops"
    );

    // Patch branch targets from block index to entry pc.
    let patch = |t: &mut u32| *t = block_pc[*t as usize];
    for op in &mut ops {
        match op {
            Op::Br { target, .. } => patch(target),
            Op::BrIf {
                then_pc, else_pc, ..
            } => {
                patch(then_pc);
                patch(else_pc);
            }
            Op::BrIfCmp {
                then_pc, else_pc, ..
            } => {
                patch(then_pc);
                patch(else_pc);
            }
            Op::BrTable { arms, default, .. } => {
                for (_, t) in arms.iter_mut() {
                    patch(t);
                }
                patch(&mut default.1);
            }
            _ => {}
        }
    }
    Some(Program {
        ops,
        nslots,
        src: src.into_boxed_slice(),
    })
}

fn compile_inst(inst: &Inst, dst: u32, block_base: u32, g: &impl Fn(u32) -> u32) -> Option<Op> {
    Some(match inst {
        Inst::ConstI32(c) => Op::Const {
            dst,
            val: Reg::from_i32(*c),
        },
        Inst::ConstI64(c) => Op::Const {
            dst,
            val: Reg::from_i64(*c),
        },
        Inst::ConstF32(b) => Op::Const {
            dst,
            val: Reg::from_f32(f32::from_bits(*b)),
        },
        Inst::ConstF64(b) => Op::Const {
            dst,
            val: Reg::from_f64(f64::from_bits(*b)),
        },
        Inst::IntBin { ty, op, a, b } => Op::IntBin {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::IntCmp { ty, op, a, b } => Op::IntCmp {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::IntUn { ty, op, a } => Op::IntUn {
            dst,
            a: g(*a),
            ty: *ty,
            op: *op,
        },
        Inst::Eqz { ty, a } => Op::Eqz {
            dst,
            a: g(*a),
            ty: *ty,
        },
        Inst::Convert { op, a } => Op::Convert {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::Select { cond, a, b } => Op::Select {
            dst,
            cond: g(*cond),
            a: g(*a),
            b: g(*b),
        },
        Inst::FBin { ty, op, a, b } => Op::FBin {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::FUn { ty, op, a } => Op::FUn {
            dst,
            a: g(*a),
            ty: *ty,
            op: *op,
        },
        Inst::FCmp { ty, op, a, b } => Op::FCmp {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::FToISat { op, a } => Op::FToISat {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::FToITrap { op, a } => Op::FToITrap {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::IToFConv { op, a } => Op::IToFConv {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::Cast { op, a } => Op::Cast {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::RefFunc { func } => Op::RefFunc { dst, func: *func },
        Inst::Load {
            op, addr, offset, ..
        } => Op::Load {
            dst,
            addr: g(*addr),
            op: *op,
            offset: *offset,
        },
        Inst::Store {
            op,
            addr,
            value,
            offset,
            ..
        } => Op::Store {
            addr: g(*addr),
            value: g(*value),
            op: *op,
            offset: *offset,
        },
        Inst::MemCopy { dst, src, len } => Op::MemCopy {
            dst: g(*dst),
            src: g(*src),
            len: g(*len),
        },
        Inst::MemMove { dst, src, len } => Op::MemMove {
            dst: g(*dst),
            src: g(*src),
            len: g(*len),
        },
        Inst::MemFill { dst, val, len } => Op::MemFill {
            dst: g(*dst),
            val: g(*val),
            len: g(*len),
        },
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => Op::AtomicLoad {
            dst,
            addr: g(*addr),
            ty: *ty,
            offset: *offset,
        },
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
            ..
        } => Op::AtomicStore {
            addr: g(*addr),
            value: g(*value),
            ty: *ty,
            offset: *offset,
        },
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
            ..
        } => Op::AtomicRmw {
            dst,
            addr: g(*addr),
            value: g(*value),
            ty: *ty,
            op: *op,
            offset: *offset,
        },
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            ..
        } => Op::AtomicCmpxchg {
            dst,
            addr: g(*addr),
            expected: g(*expected),
            replacement: g(*replacement),
            ty: *ty,
            offset: *offset,
        },
        Inst::Call { func, args } => Op::Call {
            callee: *func,
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
        },
        // `call_indirect` through module 0's natural table — self-contained (no install/invoke),
        // so the compile-time signature table resolves it. Cross-module units (install/invoke) are
        // still a later slice; here every reachable slot is a module-0 function.
        Inst::CallIndirect { ty, idx, args } => Op::CallIndirect {
            idx: g(*idx),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            want_params: ty.params.clone().into(),
            want_results: ty.results.clone().into(),
        },
        // Synchronous capability call: the generic powerbox path (guest suspended, host computes,
        // same activation continues) is driven here via `host.cap_dispatch_slots`. The
        // executor/fiber capability variants — `Instantiator` (child vCPUs), `Yielder` (co-fiber
        // yield), `JIT` (install/uninstall/invoke), and `SharedRegion` op 4 (`grant` into a child) —
        // need seams a later slice drives, so reject those (fall back to the tree-walker). These are
        // exactly the `type_id`/`op` combinations `run_inner` matches in dedicated arms ahead of its
        // generic `CapCall` arm.
        Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } => {
            use super::cap_id;
            match (*type_id, *op) {
                // §14 executor children — instantiate (op 0) spawns a confined child on the scheduler;
                // join (op 1) parks until it finishes, reusing the §12 thread join machinery (children
                // share the `threads` handle namespace). The separate-module / demand variants (5/6/7
                // and op 4) and the JIT / SharedRegion-grant variants need seams this slice doesn't
                // drive: reject (fall back).
                (cap_id::INSTANTIATOR, 0) if args.len() >= 4 => Op::Instantiate {
                    handle: g(*handle),
                    entry: g(args[0]),
                    off: g(args[1]),
                    size_log2: g(args[2]),
                    quota: g(args[3]),
                    dst,
                    grants: None,
                },
                (cap_id::INSTANTIATOR, 1) if !args.is_empty() => Op::InstJoin {
                    handle: g(*handle),
                    child: g(args[0]),
                    dst,
                },
                // op 5 = instantiate_module: the first arg is the granted `Module` handle; the carve
                // args (entry/off/size_log2/quota) follow. (join, op 1, serves both kinds.)
                (cap_id::INSTANTIATOR, 5) if args.len() >= 5 => Op::InstantiateModule {
                    handle: g(*handle),
                    module: g(args[0]),
                    entry: g(args[1]),
                    off: g(args[2]),
                    size_log2: g(args[3]),
                    quota: g(args[4]),
                    dst,
                    grants: None,
                },
                // op 13 = instantiate_module_named: op 5 + a by-name grant list. Args:
                // (module, grants_ptr, grants_n, entry, off, size_log2, quota). The driver reads the
                // grant records from the parent window and re-grants each cap into the child powerbox.
                (cap_id::INSTANTIATOR, 13) if args.len() >= 7 => Op::InstantiateModule {
                    handle: g(*handle),
                    module: g(args[0]),
                    entry: g(args[3]),
                    off: g(args[4]),
                    size_log2: g(args[5]),
                    quota: g(args[6]),
                    dst,
                    grants: Some((g(args[1]), g(args[2]))),
                },
                // CONSOLIDATION.md §3d — instantiate_rec (op 17): the record pointer is the one
                // arg; every other spawn parameter is data in the record, read at exec time.
                (cap_id::INSTANTIATOR, 17) if !args.is_empty() => Op::InstantiateRec {
                    handle: g(*handle),
                    rec: g(args[0]),
                    dst,
                },
                // §3.6 (I36 slice 2) — child_offer (op 14): mint a live-callee offer over a running
                // child's export. The mint needs the child's live env, so the op surfaces to the
                // driver; the compile only marshals `(child, export)`.
                (cap_id::INSTANTIATOR, 14) if args.len() >= 2 => Op::ChildOffer {
                    handle: g(*handle),
                    child: g(args[0]),
                    export: g(args[1]),
                    dst,
                },
                // §22 guest-driven JIT units: install/uninstall drive the dispatch table; compile /
                // compile_linked (ops 0/5) are pure host ops, so they fall through to the generic
                // dispatch below. `invoke` (op 1) is the next slice — reject it for now (fall back).
                (cap_id::JIT, 3) if !args.is_empty() => Op::JitInstall {
                    handle: g(*handle),
                    code: g(args[0]),
                    dst,
                },
                (cap_id::JIT, 4) if !args.is_empty() => Op::JitUninstall {
                    handle: g(*handle),
                    slot: g(args[0]),
                    dst,
                },
                (cap_id::JIT, 1) if !args.is_empty() => Op::JitInvoke {
                    handle: g(*handle),
                    code: g(args[0]),
                    args: args[1..].iter().map(|a| g(*a)).collect(),
                    dst,
                    // The cap.call sig is `(i64 code, params…) -> (results…)`; the unit entry's
                    // params are sig.params without the leading code-handle.
                    params: sig.params.get(1..).unwrap_or(&[]).to_vec().into(),
                    results: sig.results.clone().into(),
                },
                (cap_id::INSTANTIATOR, _) => return None,
                (cap_id::SHARED_REGION, 4) => return None,
                // §3.6 service points (I36 slice 1): `svc.poll` with the canonical one-result
                // shape compiles to the native serve-loop-core op — the module-level veto in
                // [`compile_module`] guarantees its handlers cannot park mid-dispatch, so the
                // rewind linkage runs each one to completion (or trap). `svc.wait`'s
                // empty-queue park needs a waker topology (cross-domain callers, timers) the
                // cooperative scheduler doesn't host yet, and a no-result `svc.poll` would
                // leave the op without its result-slot scratch — both still decline, falling
                // the whole module back to the tree-walk oracle, which serves.
                // (The timed `svc.wait` form — op 10 with the optional timeout arg — is
                // oracle-only and declines below; `serve_qualifies` already vetoed the module.)
                (svm_ir::CAP_SELF_TYPE_ID, op @ (9 | 10))
                    if sig.results.len() == 1 && args.is_empty() =>
                {
                    Op::SvcPoll {
                        dst,
                        wait: op == 10,
                    }
                }
                (svm_ir::CAP_SELF_TYPE_ID, 9 | 10) => return None,
                // FORK.md §9.2 — `clone_caller` (op 11) / `reap` (op 12): compiled to the native fork
                // ops (the module reached here only via the [`Seams::bytecode_serves_fork`] escape).
                // The reply/pid args are register operands, resolved in the driver.
                (svm_ir::CAP_SELF_TYPE_ID, 11) => Op::CloneCaller {
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                    has_result: !sig.results.is_empty(),
                },
                (svm_ir::CAP_SELF_TYPE_ID, 12) => Op::Reap {
                    pid: args.first().map(|a| g(*a)),
                    dst,
                    has_result: !sig.results.is_empty(),
                },
                // CALLS.md §10.6 — `fuel.remaining` (op 13) reads the vCPU's live fuel counter, which
                // the host-side `cap_dispatch_slots` can't see; rather than add a native bytecode op,
                // decline the module so it falls back to the tree-walker, which services op 13
                // directly. (The JIT does lower it inline — it owns the fuel cell's address.)
                (svm_ir::CAP_SELF_TYPE_ID, 13) => return None,
                // FORK.md §8.6 — `exec_module` (`execve` image-replace, op 14) is eval-loop-only:
                // the tree-walker folds it to `Step::Exec` before any host dispatch, so a fast tier
                // that ran the generic `cap.call` thunk would answer `-EINVAL` where the oracle
                // image-replaces — a silent divergence (INVARIANTS.md #9). Decline the module so it
                // folds to the oracle (fail-closed), as op 13 does. (OPS_PARITY.md `exec`.)
                (svm_ir::CAP_SELF_TYPE_ID, 14) => return None,
                // Generic synchronous powerbox dispatch (Stream/Clock/Memory/host-fn/JIT compile/…).
                _ => Op::CapCall {
                    type_id: *type_id,
                    op: *op,
                    handle: g(*handle),
                    params: sig.params.clone().into(),
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                    results: sig.results.clone().into(),
                },
            }
        }
        // §7/§6 reflection `cap.self.count`/`get`/`resolve`/`label`/`attest` reach here as their
        // `cap.call CAP_SELF op 0/1/2/3/4` form and compile via the generic `Op::CapCall` fallthrough
        // in the `Inst::CapCall` arm above.
        // §12 fibers — cooperative continuation switching, driven by the bytecode driver (no M:N
        // pool, no DPOR; single-vCPU). `cont.new` registers a pending fiber, `cont.resume` switches
        // in (two results), `suspend` switches back (one result).
        Inst::ContNew { func, sp } => Op::ContNew {
            func: g(*func),
            sp: g(*sp),
            dst,
        },
        // I48 — the `block: true` form idles the resumer's task on the fiber's event (see the
        // cooperative driver's `Outcome::ContResume` / fiber-park arms + `TaskState::BlockedOnFiber`)
        // instead of spinning the poll. Advisory still holds: `FIBER_PARKED` remains a legal
        // transient (the guest keeps its loop), so the deterministic explorer and any non-idling
        // path stay conforming (invariant 9).
        Inst::ContResume { k, arg, block } => Op::ContResume {
            k: g(*k),
            arg: g(*arg),
            dst,
            blocking: *block,
        },
        Inst::Suspend { value } => Op::Suspend {
            value: g(*value),
            dst,
        },
        // `<setjmp.h>` non-local jump — intra-vCPU (no scheduler escape). `setjmp` checkpoints the
        // activation's resume point (the flat per-function register layout keeps each block's slots
        // distinct, so the `setjmp` block's values survive a deeper call — no window snapshot needed,
        // unlike the tree-walker's per-block `vals`); `longjmp` pops the activation stack back to it.
        Inst::SetJmp { buf } => Op::SetJmp { buf: g(*buf), dst },
        Inst::LongJmp { buf, val } => Op::LongJmp {
            buf: g(*buf),
            val: g(*val),
        },
        // §12 threads / futex — cooperative multi-vCPU, serviced by the `drive` scheduler. (A module
        // mixing threads *and* fibers is rejected at the module level — see `compile_module` — until
        // the run-shared fiber registry / migration lands.)
        Inst::ThreadSpawn { func, sp, arg } => Op::ThreadSpawn {
            func: *func,
            sp: g(*sp),
            arg: g(*arg),
            dst,
        },
        Inst::ThreadJoin { handle } => Op::ThreadJoin {
            handle: g(*handle),
            dst,
        },
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => Op::MemoryWait {
            ty: *ty,
            addr: g(*addr),
            expected: g(*expected),
            timeout: g(*timeout),
            dst,
        },
        Inst::MemoryNotify { addr, count } => Op::MemoryNotify {
            addr: g(*addr),
            count: g(*count),
            dst,
        },
        // Cross-module / GC ops this slice doesn't drive (dispatch table / root scan) — fall back.
        // §GC conservative root enumeration — driven by the scheduler (it scans the whole vCPU
        // continuation). `call.import` must already be resolved to a `cap.call`, so it never reaches
        // a backend (a leftover is a fall-back).
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => Op::GcRoots {
            lo: g(*heap_lo),
            hi: g(*heap_hi),
            mask: g(*mask),
            buf: g(*buf),
            cap: g(*cap),
            dst,
        },
        // §7 executable named import (IMPORTS.md phase 1): lower to the **generic** cap dispatch
        // with the reserved [`svm_ir::CAP_IMPORT_TYPE_ID`] and the import index as the op — the
        // host's dispatch translates it through the instantiation-time binding table, exactly as
        // the tree-walker and the JIT thunk do (one shared implementation, three backends in
        // lockstep). No handle operand since v8 (the binding carries the granted handle); the
        // dispatch never read one, so the register is simply absent.
        Inst::CallImport {
            import,
            op,
            sig,
            args,
        } => Op::CapCall {
            type_id: svm_ir::CAP_IMPORT_TYPE_ID,
            // §3.5: the reserved import dispatch packs `(slot | consumer_op << 16)`.
            op: *import | (*op << 16),
            handle: u32::MAX, // no operand (v8); the exec passes 0, the dispatch ignores it
            params: sig.params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: sig.results.clone().into(),
        },
        // §7/§22 symbolic call: when bound at instantiation it is a flat import dispatch
        // (op 0); the legacy handle operand is a live register the dispatch ignores.
        Inst::CallSym {
            import, sig, args, ..
        } => Op::CapCall {
            type_id: svm_ir::CAP_IMPORT_TYPE_ID,
            op: *import,
            handle: u32::MAX,
            params: sig.params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: sig.results.clone().into(),
        },
        // §3.5 dynamic-mode dispatch by type-section reference: the reserved dyn entry packs
        // `(type_idx | op << 16)`; the handle register is live.
        Inst::CallImportDyn {
            ty,
            op,
            sig,
            handle,
            args,
        } => Op::CapCall {
            type_id: svm_ir::CAP_DYN_TYPE_ID,
            op: *ty | (*op << 16),
            handle: g(*handle),
            params: sig.params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: sig.results.clone().into(),
        },
        // §3.5 self-namespace extensions (see `Op::CapSelfExt`).
        Inst::ExportHandle { export } => Op::CapSelfExt {
            op: 8 | (*export << 8),
            handle: None,
            dst,
        },
        Inst::CapSelfTypeId { ty } => Op::CapSelfExt {
            op: 6 | (*ty << 8),
            handle: None,
            dst,
        },
        Inst::CapSelfCovers { handle, ty } => Op::CapSelfExt {
            op: 7 | (*ty << 8),
            handle: Some(g(*handle)),
            dst,
        },
        // Phase-2 `import.attach` (IMPORTS.md): the attach sentinel with the handle value as the
        // one argument — the same shared host entry as the tree-walker and the JIT.
        Inst::ImportAttach { import, handle } => Op::CapCall {
            type_id: svm_ir::CAP_IMPORT_ATTACH_TYPE_ID,
            op: *import,
            handle: g(*handle),
            params: [].into(),
            args: [g(*handle)].into(),
            dst,
            results: [ValType::I32].into(),
        },
        // §12.8 4A.5: serviced from the running `Vm`'s region base (the reference `eval_inst` has no
        // context), so it gets a dedicated op rather than the `Eval` fallback.
        Inst::DurableShadowBase => Op::DurableShadowBase { dst },
        // §12 per-vCPU TLS register: serviced from the running `Vm`'s `tls` word (the reference
        // `eval_inst` traps on it — no vCPU context), so it gets a dedicated op rather than `Eval`.
        Inst::VcpuTlsGet => Op::VcpuTlsGet { dst },
        Inst::VcpuTlsSet { val } => Op::VcpuTlsSet { val: g(*val) },
        // Everything else is a pure value op or a no-result store that the reference `eval_inst`
        // already implements (the SIMD/`v128`/fence long tail): delegate to it against this block's
        // sub-window, reusing the exact semantics rather than re-inlining ~30 lane ops.
        other => Op::Eval {
            inst: Box::new(other.clone()),
            block_base,
            dst,
        },
    })
}

/// Build the linear-memory window from `m`'s memory declaration + data segments, exactly like
/// [`crate::run`] (a module with no memory yields `None`).
fn build_mem(m: &Module) -> Option<Mem> {
    m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data);
        mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        mm
    })
}

/// Compile `m`'s function `func` and run it on the bytecode engine, or `None` if it (or any
/// function it can reach by direct call) uses an op outside this slice's subset. Builds a fresh
/// linear-memory window from `m`'s memory declaration + data segments, exactly like
/// [`crate::run`]. Returns typed result `Value`s. The equality harness compares this to `run`.
pub fn compile_and_run(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
) -> Option<Result<Vec<Value>, Trap>> {
    // No capabilities granted: an empty powerbox (any `cap.call` is inert → `CapFault`), exactly
    // like [`crate::run`], so this stays a faithful mirror for the equality harness.
    let mut host = Host::new();
    compile_and_run_with_host(m, func, args, fuel, &mut host)
}

/// Host-carrying [`compile_and_run`]: the powerbox is live, so synchronous capability calls
/// (`cap.call` through the generic dispatch) execute against it. `None` if the module uses an op
/// outside this slice's subset (including the executor/fiber capability variants) — the caller
/// (`crate::run_with_host_fast`) then falls back to the tree-walker.
pub fn compile_and_run_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    // Size the dispatch table to the granted `Jit` table reservation (matching the tree-walker's
    // `DomainTable::new(funcs, jit_table_log2)`), so guest-driven `install` returns the same slots.
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = build_mem(m);
    Some(run(dom, func, args, fuel, &mut mem, host))
}

/// What [`compile_and_run_with_host_traced`] returns — the shared traced-run shape (result + trap-time
/// backtrace + trapping fiber). The single-step path is root-only, so its fiber is `-1` (a trap) or
/// `None` (clean); a fibered run is a seam it declines, so the tree-walker reports the real handle.
pub type TracedRun = super::TracedRun;

/// Trap-time-backtrace counterpart of [`compile_and_run_with_host`] — the bytecode mirror of the
/// tree-walker's [`crate::run_with_host_traced`]. Drives the entry **one op at a time** (the proven
/// single-vCPU debug seam, as [`ir_trace`] does — `budget = 1` is bit-identical to run-to-completion,
/// INTERP_PERF.md Slice 1c-2) so that on a trap the `Vm`'s reified continuation still points at the
/// faulting op (the `Err` path never writes the cursor back) and its caller windows are intact; the
/// backtrace is then read off that continuation by [`vm_trap_bt`] — the flat-window analogue of the
/// tree-walker snapshotting `v.frames`. Returns `(result, backtrace)` (innermost frame first, as
/// [`crate::IrPc`]s; empty on a clean finish), resolvable to source with [`crate::source_loc`].
///
/// `None` (caller falls back to [`crate::run_with_host_traced`]) when the module is outside the
/// engine's subset, **or** when a step reaches a concurrency/coroutine seam — backtraces are
/// single-vCPU, seam-free scope (DEBUGGING.md S4), exactly like [`ir_trace`]. Single-stepping is a
/// cold diagnostic path, so the per-op suspend/resume overhead never touches the production
/// `run_fast` loop.
pub fn compile_and_run_with_host_traced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Option<TracedRun> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new(), None));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = build_mem(m);
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Err(e), Vec::new(), None)),
    };
    loop {
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut *host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((Ok(vals), Vec::new(), None)),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope (fall back to tree-walker)
            Err(t) => {
                let bt = vm_trap_bt(&vm, &dom.source, &t);
                // This single-step path only ever drives the **root** (a fiber/thread op is a seam →
                // the `Ok(_)` arm above bails to the tree-walker), so a trap here is always the root —
                // attributed `-1`, matching the JIT's root-trap convention.
                return Some((Err(t), bt, Some(-1)));
            }
        }
    }
}

/// The trap-time backtrace of a `Vm` paused (by an `Err` from [`Vm::resume`]) on a faulting op:
/// the [`crate::IrPc`] of every live activation, **innermost frame first** — the flat-window analogue
/// of the tree-walker's [`crate::frames_to_pcs`] over `Vec<Frame>`. The cursor (`module`/`cur`/`pc`)
/// is the trapping op (the `Err` path leaves it as the prior op-boundary persisted it).
///
/// **Cursor-advance parity with the tree-walker** (`run_inner`): the tree-walker charges fuel, then
/// does `inst += 1`, then evaluates the op — so the live frame's recorded `inst` is one *past* the op
/// for any trap raised in evaluation (memory fault, div-by-zero, malformed, …), but the op *itself*
/// for an [`Trap::OutOfFuel`] (caught before the advance). The bytecode loop instead leaves `pc` on
/// the trapping op for *both*, so to report identical `IrPc`s we add `1` to the innermost frame's
/// `inst` unless the trap is `OutOfFuel`. Every suspended caller in `stack` already resumes at
/// `call_pc + 1` (the tree-walker likewise advances a caller's `inst` past the call before
/// descending), so its call op sits at `resume_pc - 1` and we report `inst + 1` for it. `None`-`src`
/// ops (terminators) are skipped, matching [`Program::src`] / [`Vm::cur_ir_pc`].
fn vm_trap_bt(vm: &Vm, source: &ModuleSource, trap: &Trap) -> Vec<super::IrPc> {
    let mut bt = Vec::new();
    let Some(c) = source.get(vm.module) else {
        return bt;
    };
    if let Some((block, inst)) = c.progs[vm.cur].src.get(vm.pc).copied().flatten() {
        // An instruction's recorded `inst` advances past the op exactly when the tree-walker's did
        // (it does `inst += 1` before evaluating, so every trap but `OutOfFuel` lands one past); a
        // terminator (`unreachable`, `return_call_indirect`) is already stored as `insts.len()`, the
        // exact `inst` the tree-walker's frame carries there, and gets no bump.
        let inst = if inst & SRC_TERM != 0 {
            (inst & !SRC_TERM) as usize
        } else {
            inst as usize + !matches!(trap, Trap::OutOfFuel) as usize
        };
        bt.push(super::IrPc {
            module: vm.module as u32,
            func: vm.cur as FuncIdx,
            block: block as usize,
            inst,
        });
    }
    // Each suspended caller resumes at `call_pc + 1` (a call is an instruction, never a terminator),
    // so its call op sits at `resume_pc - 1`; report `inst + 1`, mirroring the tree-walker advancing a
    // caller's `inst` past the call before descending.
    for &(module, prog, _base, resume_pc, _ret) in vm.stack.iter().rev() {
        let call_pc = resume_pc.wrapping_sub(1);
        let Some(cm) = source.get(module) else {
            continue;
        };
        if let Some((block, inst)) = cm.progs[prog].src.get(call_pc).copied().flatten() {
            bt.push(super::IrPc {
                module: module as u32,
                func: prog as FuncIdx,
                block: block as usize,
                inst: (inst & !SRC_TERM) as usize + 1,
            });
        }
    }
    bt
}

/// A run result paired with the final window snapshot (the low `init_mem.len()` bytes).
pub type Capture = (Result<Vec<Value>, Trap>, Vec<u8>);

/// Like [`compile_and_run`], but **seeds** the window with `init_mem` first and returns the final
/// window snapshot (the low `init_mem.len()` bytes) alongside the result — the bytecode mirror of
/// [`crate::run_capture_reserved`]. Used by `bytecode_gc_roots.rs` to read back the roots buffer for
/// the §GC soundness check. `None` if the module is outside the engine's subset.
pub fn compile_and_run_capture(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

/// Like [`compile_and_run_capture`], but the guest window is backed by a **caller-provided**
/// [`Region`] (a `Region::shared` over host memory) rather than an engine-`mmap`ped one — the
/// substrate→engine bridge for the parallel-wasm backend (THREADS.md step 3). On wasm `back` spans the
/// host's shared linear memory, so the root vCPU here and the per-vCPU Workers a later step spawns all
/// execute over **one shared window**. Today still cooperative (the existing `drive`); only the
/// backing changes from owned to borrowed — so a guest's result + final image are identical to
/// [`compile_and_run_capture`], and its memory effects land in the caller's buffer. (The crate stays
/// `#![forbid(unsafe_code)]`: the `unsafe` of borrowing host memory is in the embedder's
/// `Region::shared` call that built `back`.)
pub fn compile_and_run_capture_over(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(
            DEFAULT_RESERVED_LOG2,
            mc.size_log2,
            std::sync::Arc::clone(&back),
        );
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

/// Run `func(args)` over the caller-provided shared window `back` against a caller-prepared `host`,
/// returning the typed results (`None` if the module is outside the engine's subset). Unlike
/// [`compile_and_run_capture_over`] this carries a live `host` (so `cap.call`s execute) and — when
/// `seed_data` is `false` — it does **not** re-seed or re-apply the module's data segments: the window
/// in `back` is already live, so re-initialising would clobber the guest's globals/heap.
///
/// This is the browser wasm-JIT **reactor** cross-tier seam: the emitted `tick` (run by the host over
/// this same window) bounces a call to a non-emitted function here — the callee runs on the
/// interpreter over the shared window, its memory effects landing in the bytes the emitted code reads.
/// Pass `seed_data = true` exactly once, for the initial `_start`, to data-initialise the window before
/// the first frame; every per-frame cross-tier callee passes `false`.
pub fn compile_and_run_over_shared_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    back: std::sync::Arc<super::Region>,
    host: &mut Host,
    seed_data: bool,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, mc.size_log2, back);
        if seed_data {
            mm.init_data(&m.data);
            mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        }
        mm
    });
    Some(run(dom, func, args, fuel, &mut mem, host))
}

/// A module compiled **once** for repeated runs over a caller-provided shared window — the cached form
/// of [`compile_and_run_over_shared_with_host`]. The browser wasm-JIT reactor bounces a handful of
/// interpreter helpers per frame through `env.call_interp`; recompiling the whole module on every bounce
/// (as the one-shot does) dominates the frame — for Doom, ~6 ms × 3 calls ≈ 19 ms of a 20 ms frame. This
/// holds the compiled source (a cheap `Arc` clone seeds each run's throwaway [`Domain`]) so a cross-tier
/// run is just build-window + interpret, like [`Reactor`] but over the caller's shared window.
pub struct SharedProgram {
    source: std::sync::Arc<ModuleSource>,
    n_funcs: usize,
    mem_size_log2: Option<u8>,
    data: Vec<super::Data>,
    /// #964: the module's NULL-guard extent (`0` = unmarked/legacy).
    null_guard: u64,
}

impl SharedProgram {
    /// Compile `m` once (`None` if it uses an op outside the engine's subset).
    pub fn compile(m: &Module) -> Option<SharedProgram> {
        let c = compile_module_for(m)?;
        let n_funcs = c.progs.len();
        Some(SharedProgram {
            source: std::sync::Arc::new(ModuleSource::new(c)),
            n_funcs,
            mem_size_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone(),
            null_guard: svm_ir::module_null_guard(m).unwrap_or(0),
        })
    }

    /// Run `func(args)` over the shared window `back` with `host`, **without recompiling**. `seed_data`
    /// applies the module's data segments first — pass `true` exactly once (the initial `_start`), and
    /// `false` for every per-frame cross-tier callee (the window in `back` is already live). `Err` on a
    /// trap (`Exit` surfaces as `Trap::Exit`), or `Trap::Malformed` if `func` is out of range.
    pub fn run_over(
        &self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        back: std::sync::Arc<super::Region>,
        host: &mut Host,
        seed_data: bool,
    ) -> Result<Vec<Value>, Trap> {
        self.run_over_grown(
            func,
            args,
            fuel,
            back,
            host,
            seed_data,
            DEFAULT_RESERVED_LOG2,
            None,
        )
        .0
    }

    /// [`run_over`](Self::run_over) for a **restorable warm session** (#816): the reservation is
    /// caller-chosen (clamp it to the shared backing's size — a reservation past the backing lets
    /// guest writes silently vanish instead of failing the `map`), and a captured **explicit
    /// page-state map** can be re-established before the run: `prots = Some(entries)` re-inserts
    /// each `(byte offset, kind)` entry (the [`Mem::map_info`] encoding — the on-ramp's
    /// `protect`ed rodata and the `vm_map`-grown tail alike) *without zeroing* the pages the
    /// caller already restored ([`Mem::seed_pages`]) — so a page-managing warm image survives the
    /// fresh-`Mem`-per-call shape. Returns the run's result plus the post-run explicit page map:
    /// `Some(entries)` to seed the next call with (empty for a plain flat window), or `None` if
    /// the guest aliased a §13 `SharedRegion` page — a byte restore cannot reproduce an alias, so
    /// a warm driver must fail closed on it. `Some(vec![])` for a memory-less module.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_over_grown(
        &self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        back: std::sync::Arc<super::Region>,
        host: &mut Host,
        seed_data: bool,
        reserved_log2: u8,
        prots: Option<&[(u64, u8)]>,
    ) -> (Result<Vec<Value>, Trap>, Option<Vec<(u64, u8)>>) {
        if func as usize >= self.n_funcs {
            return (Err(Trap::Malformed), None);
        }
        // A fresh natural dispatch table over the shared compiled source (cheap: an `Arc` clone + the
        // slot vector) — the cross-tier reactor carries no §22 install state between calls.
        let dom = Domain::child(self.source.clone(), SharedSlots::new(self.n_funcs, 0, 0));
        let mut mem = self.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(reserved_log2, sl, back);
            if seed_data {
                mm.init_data(&self.data);
            }
            mm.seed_null_guard(self.null_guard); // #964
            if let Some(entries) = prots {
                mm.seed_pages(entries);
            }
            mm
        });
        let out = run(dom, func, args, fuel, &mut mem, host);
        let pages = match mem.as_ref() {
            None => Some(Vec::new()),
            Some(m) => {
                let (_, _, _, entries) = m.map_info();
                if entries.iter().any(|&(_, kind)| kind == 3) {
                    None // §13 Backed alias — unrestorable by a byte snapshot; fail closed
                } else {
                    Some(entries)
                }
            }
        };
        (out, pages)
    }
}

/// THREADS.md step 4c — the **parallel** sibling of [`compile_and_run_capture_over`]: run the guest's
/// `thread.spawn`ed vCPUs on **separate OS threads** (the native stand-in for per-vCPU wasm Workers)
/// over the **one** caller-owned shared window, instead of cooperatively multiplexing them onto one
/// thread. Every vCPU executes over the same `Region::shared` backing — `thread.spawn`/`join` +
/// hardware `atomic.*` are genuine cross-core operations, not a single-thread interleaving. This is
/// the host-selected `Parallel` mode; the cooperative [`compile_and_run_capture_over`] is its
/// **deterministic oracle** (differential-tested in `bytecode_parallel.rs`).
///
/// Scope: the **full threads model** — `thread.spawn`/`join`, the `memory.wait`/`notify` futex
/// (a genuine cross-thread [`Futex`], not a single-thread park queue), and atomics — plus pure compute.
/// The `Domain` is shared `&`-immutably across threads, so the two events that need a `&mut
/// Domain`/shared powerbox — §14 `instantiate` and §22 JIT install — **fail closed**
/// (`Trap::ThreadFault`) here rather than run wrong; they are the remaining follow-ons. Returns `None`
/// only if the module is outside the engine's subset, same as the cooperative entry.
pub fn compile_and_run_capture_over_parallel(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    compile_and_run_capture_over_parallel_with_host(m, func, args, fuel, init_mem, back, &mut host)
}

/// Like [`compile_and_run_capture_over_parallel`], but runs over a **caller-prepared `host`** (the
/// powerbox) shared by every parallel vCPU (THREADS.md 4c-host). A spawned vCPU's `cap.call` dispatches
/// on the **same** host as the root, serialized per call by an internal lock — so host I/O from worker
/// vCPUs works, with compute/atomics/futex still fully parallel. Determinism note: this is the **opt-in
/// parallel** mode, so stateful-cap interleaving (e.g. `Clock.now` values, the order of distinct
/// `stdout` writes) races as real threads do; the **cooperative** entries remain the deterministic
/// oracle. The caller reads the host back (its `stdout`/state) after the run.
pub fn compile_and_run_capture_over_parallel_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
    host: &mut Host,
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(
            DEFAULT_RESERVED_LOG2,
            mc.size_log2,
            std::sync::Arc::clone(&back),
        );
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        mm
    });
    let (r, mem) = drive_parallel(dom, func, args, *fuel, mem, host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

// === THREADS.md step 4c-wasm — the resumable per-vCPU primitive ==================================
// `drive_parallel` runs a guest's vCPUs on native OS threads it spawns itself. The browser can't:
// wasm32 has no `thread::spawn`, so a guest `thread.spawn` must bubble out to JS, which creates a
// Worker that re-enters the engine to run that one vCPU. That needs a *resumable, single-vCPU* entry
// the **host** orchestrates — pausing on each multi-vCPU event (`thread.spawn`/`join`,
// `memory.wait`/`notify`) and resuming once the host has serviced it. `Program` + `Vcpu` are exactly
// that primitive (platform-agnostic, no threads, no FFI): the wasm embedder drives them across Workers
// with the real `memory.atomic.wait`/`notify` futex, and the native orchestration test drives them
// across `std::thread`s as the differential proof.

/// A compiled module, shareable **read-only** across vCPUs / threads / Workers (its [`Domain`] is
/// `Sync`). Built once per run; each [`Vcpu`] borrows it. Also carries the memory declaration + data
/// segments so each vCPU can build its window over the shared backing.
pub struct VcpuProgram {
    dom: Domain,
    mem_size_log2: Option<u8>,
    data: Vec<svm_ir::Data>,
    /// #964: the module's NULL-guard extent (`0` = unmarked/legacy), captured at compile so every
    /// window this program is run over seeds the same guard the module's layout was built for.
    null_guard: u64,
}

impl VcpuProgram {
    /// Compile `m` for the bytecode engine, or `None` if it uses an op outside the engine's subset.
    /// The dispatch table is natural-sized (no §22 `install` room); use [`compile_with_jit_table`] to
    /// reserve padding slots for guest-driven install.
    ///
    /// [`compile_with_jit_table`]: VcpuProgram::compile_with_jit_table
    pub fn compile(m: &Module) -> Option<VcpuProgram> {
        Self::compile_with_jit_table(m, 0)
    }

    /// Like [`compile`](VcpuProgram::compile), but reserve a `call_indirect` table of `2^table_log2`
    /// slots for §22 `Jit.install` — pass the **same** value the embedder gave `grant_jit_with_table`
    /// (the powerbox's [`Host::jit_table_log2`]), so guest-driven install lands at the same slots the
    /// cooperative oracle uses. `0` ⇒ natural size (no install room).
    pub fn compile_with_jit_table(m: &Module, table_log2: u8) -> Option<VcpuProgram> {
        let c = compile_module_for(m)?;
        let dom = Domain::new(c, table_log2);
        Some(VcpuProgram {
            dom,
            mem_size_log2: m.memory.as_ref().map(|mc| mc.size_log2),
            data: m.data.clone(),
            null_guard: svm_ir::module_null_guard(m).unwrap_or(0),
        })
    }

    /// Number of functions (a `thread.spawn` target is bounds-checked against this).
    pub fn func_count(&self) -> usize {
        self.dom.source.primary().progs.len()
    }
}

/// A persistent, single-vCPU **reactor instance** — the "instantiate once, call exports many times"
/// shape with **full-memory** fidelity. Unlike the snapshot reactors (`svm-run`'s `Session`, the
/// browser `OnrampReactor`), which round-trip only a fixed low prefix and so lose a `vm_map`-grown
/// heap between calls, a `Reactor` keeps the guest's linear-memory window **live** across calls:
/// globals, BSS, **and** the grown heap all persist frame-to-frame because the window is never torn
/// down. Host capabilities are serviced inline (the [`run`]-to-completion model, identical to
/// [`compile_and_run_with_host`]), so I/O guests work — `stdout`, and the `display`/`keyboard` caps
/// the interactive playground guests (the Doom path) use.
///
/// Single-vCPU: a guest that `thread.spawn`s is out of scope (the live window is not shared with other
/// vCPUs — use the multi-worker `drive` path for those). The usual shape is `open` → `call(0, …)` to
/// run the on-ramp `_start` bootstrap once, then `call(tick, …)` once per frame.
pub struct Reactor {
    /// The compiled program, shared (an `Arc` clone seeds each call's throwaway `Domain`).
    source: std::sync::Arc<ModuleSource>,
    n_funcs: usize,
    /// The live guest window — retained across calls (this is the whole point). `None` for a
    /// memory-less module.
    mem: Option<Mem>,
}

impl Reactor {
    /// Open a reactor over a freshly compiled `m` (`None` if `m` uses an op outside the engine's
    /// subset): build the guest window once (its data segments applied) and keep it live.
    pub fn open(m: &Module) -> Option<Reactor> {
        let c = compile_module_for(m)?;
        let n_funcs = c.progs.len();
        Some(Reactor {
            source: std::sync::Arc::new(ModuleSource::new(c)),
            n_funcs,
            mem: build_mem(m),
        })
    }

    /// Call `func(args)` on the **live** window, servicing host caps inline; the window (including a
    /// grown heap) persists after the call. `Err` on a trap (an `Exit` surfaces as `Trap::Exit`), or
    /// `Trap::Malformed` if `func` is out of range.
    pub fn call(
        &mut self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        host: &mut Host,
    ) -> Result<Vec<Value>, Trap> {
        if func as usize >= self.n_funcs {
            return Err(Trap::Malformed);
        }
        // A fresh natural dispatch table over the shared compiled source (cheap: an `Arc` clone + the
        // slot vector) — there is no §22 install state to carry between frames, so a natural table each
        // call is correct. `run` consumes the `Domain`; the persistent `mem` carries state across calls.
        let dom = Domain::child(self.source.clone(), SharedSlots::new(self.n_funcs, 0, 0));
        run(dom, func, args, fuel, &mut self.mem, host)
    }
}

/// A persistent single-vCPU reactor driven through the **resumable [`Vcpu`]** — the vehicle the
/// browser wasm-JIT **tier-up** rides (BROWSER.md § "wasm-JIT tier"). Like [`Reactor`], it keeps the
/// guest window live across frames (globals, BSS, and the `vm_map`-grown heap, with its address-space
/// commit state), but each frame runs on a `Vcpu` instead of the one-shot [`run`]: a direct `Call` to
/// a [`with_jit_eligible`](Vcpu::with_jit_eligible) function surfaces as a [`VcpuEvent::TierUp`] the
/// caller services (the browser runs the emitted `f{func}` on the raw window; a native driver runs the
/// callee on the interpreter) instead of interpreting it. With no eligibility set it is a faithful,
/// interpreter-only substitute for [`Reactor`] — the differential the reactor tests assert.
///
/// The window lives in the caller-provided `back` [`Region`] (a `Region::shared` over the host's
/// linear memory in the browser; a leaked buffer natively), sized to hold the guest's grown heap. The
/// `Host` is shared (a `Mutex<Host>`) so its capabilities — `display`/`keyboard`/`fs`, stdout —
/// persist across frames and are serviced inline during each frame's `cap.call`s.
pub struct VcpuReactor {
    prog: VcpuProgram,
    /// The live window, carried across per-frame vCPUs via [`Vcpu::take_mem`]. `None` only for a
    /// memory-less module.
    mem: Option<Mem>,
    /// The tier-up eligibility bitmap (`None` ⇒ everything interprets — the pure-substitute mode).
    eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 **paged tier-up**: the eligible set was emitted with the software page-check
    /// (`compile_module_tierup_paged`). Frames then surface tier-up regardless of the window's
    /// scalar representability, and each `TierUp` hands `service` the live [`MemMapInfo`] to build
    /// its page-state table from ([`build_pagestate_table`]).
    page_checked: bool,
}

impl VcpuReactor {
    /// Open over the persistent window `back`: compile `m`, then run `_start` (func 0) once over a
    /// freshly seeded + data-initialised window to bootstrap the guest, keeping the window live for
    /// the per-frame [`frame`](VcpuReactor::frame) calls. `cap.call`s in `_start` (e.g. Doom's WAD
    /// read through `fs`) are serviced inline against `host`. `Err` if `m` is outside the engine's
    /// subset (`Malformed`) or `_start` traps.
    pub fn open(
        m: &Module,
        back: std::sync::Arc<super::Region>,
        host: &std::sync::Mutex<Host>,
        start_args: &[Value],
    ) -> Result<VcpuReactor, Trap> {
        let prog = VcpuProgram::compile(m).ok_or(Trap::Malformed)?;
        let mem;
        {
            let mut vcpu = Vcpu::new_root(&prog, 0, start_args, back, &[])?.with_shared_host(host);
            // `_start` runs to completion in one `run`: `cap.call`s are serviced inline (shared host),
            // and a reactor is single-vCPU with no tier-up during open — so no spawn/join/wait/JIT/
            // tier-up event can occur (a `thread.spawn`ing guest is out of scope).
            match vcpu.run() {
                VcpuEvent::Done(_) => {}
                VcpuEvent::Trapped(t) => return Err(t),
                _ => return Err(Trap::Malformed),
            }
            mem = vcpu.take_mem();
        }
        Ok(VcpuReactor {
            prog,
            mem,
            eligible: None,
            page_checked: false,
        })
    }

    /// Enable wasm-JIT tier-up: a direct `Call` to a function `f` with `eligible[f] == true` surfaces
    /// as [`VcpuEvent::TierUp`] for the `frame` caller to service. `None` (the default) interprets
    /// everything — the faithful [`Reactor`] substitute.
    pub fn with_jit_eligible(mut self, eligible: std::sync::Arc<[bool]>) -> VcpuReactor {
        self.eligible = Some(eligible);
        self
    }

    /// #750 paged tier-up: mark the eligible set as page-checked (see [`Vcpu::with_jit_page_checked`]).
    /// Each frame's `TierUp` then carries the live [`MemMapInfo`] to `service` so the driver can
    /// refresh its page-state table ([`build_pagestate_table`]) before running emitted code.
    pub fn with_jit_page_checked(mut self) -> VcpuReactor {
        self.page_checked = true;
        self
    }

    /// Run `func(args)` on the live window for one frame, servicing host caps inline against `host`.
    /// A [`VcpuEvent::TierUp`] is handed to `service(func, argv, mapped, map_info)` — return the
    /// callee's i64 result slots (or an `Err(Trap)` to propagate the emitted region's trap).
    /// `mapped` is the window's scalar committed extent at call entry: `service` MUST write it to
    /// the emitted module's `"mapped"` global before invoking `f{func}` (#717 host sync).
    /// `map_info` is `Some` only on a [`with_jit_page_checked`](Self::with_jit_page_checked)
    /// reactor: the live page map, from which `service` builds its page-state table
    /// ([`build_pagestate_table`]) and writes the returned coverage to `"mapped"` **instead** of
    /// the event's value (#750). With no eligibility set, `service` is never called. The window
    /// persists after the call (reclaimed for the next frame).
    pub fn frame<F>(
        &mut self,
        func: FuncIdx,
        args: &[Value],
        host: &std::sync::Mutex<Host>,
        mut service: F,
    ) -> Result<Vec<Value>, Trap>
    where
        F: FnMut(u32, &[i64], u64, Option<MemMapInfo>) -> Result<Vec<i64>, Trap>,
    {
        let mem = self.mem.take();
        let result;
        let reclaimed;
        {
            let mut vcpu =
                Vcpu::with_mem(&self.prog, func, args, mem, Host::new())?.with_shared_host(host);
            if let Some(e) = &self.eligible {
                vcpu = vcpu.with_jit_eligible(e.clone());
            }
            if self.page_checked {
                vcpu = vcpu.with_jit_page_checked();
            }
            result = loop {
                match vcpu.run() {
                    VcpuEvent::Done(v) => break Ok(v),
                    VcpuEvent::Trapped(t) => break Err(t),
                    VcpuEvent::TierUp { func, argv, mapped } => {
                        // Paged reactors snapshot the live page map for the driver's table build;
                        // computed only here (page state is frozen while emitted code runs).
                        let info = if self.page_checked {
                            vcpu.mem_map_info()
                        } else {
                            None
                        };
                        match service(func, &argv, mapped, info) {
                            Ok(vals) => vcpu.deliver_tierup(&vals),
                            Err(t) => vcpu.deliver_tierup_trap(t),
                        }
                    }
                    // Single-vCPU reactor: no spawn/join/wait/JIT-install events.
                    _ => break Err(Trap::Malformed),
                }
            };
            reclaimed = vcpu.take_mem();
        }
        self.mem = reclaimed;
        result
    }
}

/// A host-serviced pause point of a [`Vcpu`]. Everything the engine can't do alone on one thread
/// becomes one of these; the host performs the effect (spawn a Worker, futex-wait, …) and resumes the
/// vCPU with the result. Mirrors the cooperative `drive`'s `VcpuStop` arms, but handed to an external
/// orchestrator instead of serviced in-process.
pub enum VcpuEvent {
    /// The vCPU finished with these results.
    Done(Vec<Value>),
    /// The vCPU trapped (a child-join trap propagates here too).
    Trapped(Trap),
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): the interpreter reached a direct `Call`
    /// to the eligible function `func` (see [`Vcpu::with_jit_eligible`]). The host runs the emitted
    /// `f{func}(win, env, ...argv)` region on its Worker — a **top-level** call, so a guest trap is a
    /// catchable `RuntimeError` and never corrupts the engine — then calls [`Vcpu::deliver_tierup`]
    /// with the results, or [`Vcpu::deliver_tierup_trap`] if the region trapped. `argv` is the
    /// marshalled arguments as raw i64 slots (the host reads them per `func`'s signature).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        /// The window's scalar committed extent at call entry ([`Mem::scalar_extent`]) — write it to
        /// the emitted module's `"mapped"` global before invoking `f{func}` (#717 host sync).
        mapped: u64,
    },
    /// `thread.spawn`: start `func(sp, arg)` as a new vCPU, then call [`Vcpu::deliver_handle`] with the
    /// handle the guest will `join` it by (the host assigns handles densely per spawner: 0, 1, …).
    /// `module` is the spawning frame's module (0 for plain guests; an installed §22 unit's index
    /// when its code spawns) — build the child with [`Vcpu::new_child_in`] so `func` resolves there.
    Spawn {
        func: u32,
        sp: i64,
        arg: i64,
        module: u32,
    },
    /// `thread.join`: obtain child `handle`'s result, then call [`Vcpu::deliver_join`].
    Join { handle: i32 },
    /// `memory.wait`: run the futex wait on `addr`, then call [`Vcpu::deliver_code`] with the wasm code
    /// (0 = woken, 1 = not-equal, 2 = timed-out).
    Wait {
        addr: u64,
        expected: u64,
        width: u32,
        timeout: u64,
    },
    /// `memory.notify`: wake up to `count` waiters on `addr`, then call [`Vcpu::deliver_code`] with the
    /// number actually woken.
    Notify { addr: u64, count: i32 },
    /// §22 `Jit.install`: the host (which holds the powerbox) resolves authority for `handle` +
    /// code-handle `code`, returning the unit's funcs — then calls [`Vcpu::deliver_jit_install`]. The
    /// vCPU compiles + installs into the **shared** [`Domain`] (visible to every vCPU/Worker via the
    /// interior-mutable table) and writes the slot (or `-ENOSPC`) to the awaiting dst.
    JitInstall { handle: i32, code: i32 },
    /// §22 `Jit.uninstall`: the host checks authority for `handle`, then calls
    /// [`Vcpu::deliver_jit_uninstall`]; the vCPU clears the shared table `slot` (`0`/`EINVAL` → dst).
    JitUninstall { handle: i32, slot: i64 },
    /// §22 `Jit.invoke`: the host resolves the unit's funcs (authority + cross-domain), then calls
    /// [`Vcpu::deliver_jit_invoke`]; the vCPU compiles, arity-checks, and runs the unit synchronously
    /// over its window, writing the results to the awaiting dst.
    ///
    /// A **codegen** host runs the unit on emitted wasm instead ([`Vcpu::deliver_jit_invoke_vals`]).
    /// `mapped` is the window's scalar committed extent at the invoke ([`Mem::scalar_extent`]):
    /// `Some(H)` MUST be written to the emitted unit's `"mapped"` global before running it (#717
    /// host sync — same contract as [`VcpuEvent::TierUp`]); `None` means the window state is not
    /// representable by the single bound, so the host must **decline** emitted execution for this
    /// invoke and use the interpreted delivery — fail-closed, the interpreter honors the full page
    /// map. `Some(0)` for a memory-less module (nothing to bound).
    JitInvoke {
        handle: i32,
        code: i32,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: Option<u64>,
    },
    /// §14 `Instantiator.instantiate` / `instantiate_module` (THREADS.md 4c-domain §14-D2): start a
    /// **confined executor child** vCPU over the carve, then call [`Vcpu::deliver_handle`] with the
    /// handle the guest will `join` it by — exactly the [`VcpuEvent::Spawn`] protocol. All the
    /// authority-bearing work already happened in this vCPU before the event surfaced (the
    /// `Instantiator` grant resolved in-Vm; the carve validated, `-EINVAL` never surfacing; for a
    /// module child the granted `Module` resolved from this vCPU's powerbox, compiled, **pushed to the
    /// shared source**, and its data segments materialized into the carve). The host's only job is
    /// mechanical: start a Worker/thread running
    /// [`Vcpu::new_confined_child`]`(prog, module, entry, carve_region, size_log2, fuel)` over
    /// `[win + carve, win + carve + 2^size_log2)` and wire its completion slot into `join` — a
    /// confined child is just a child Worker with a shifted, smaller window (DESIGN.md §14: a
    /// sub-window is indistinguishable from a top-level window).
    Instantiate {
        /// The child's module: `0` (the primary) for `instantiate`; the pushed shared-source index
        /// for `instantiate_module`.
        module: u32,
        entry: u32,
        /// Byte offset of the carve within **this vCPU's window** (the host adds its own window
        /// base/pointer — nesting then composes with no special casing: a confined child's own
        /// `Instantiate` events are relative to *its* window).
        carve: u64,
        size_log2: u8,
        /// The child's fuel, already sub-allocated (`min(quota, parent fuel)`, or the parent's fuel
        /// when the guest passed no quota).
        fuel: u64,
    },
    /// **Blocking stdin park** (a persistent interactive session, e.g. the browser Postgres console):
    /// the guest `read` a `Stream{In}` cap whose buffer is exhausted, under [`Host::set_stdin_blocking`].
    /// The read did **not** complete (nothing written, pc un-advanced); the host pushes more bytes with
    /// [`Vcpu::push_stdin`] and calls [`run`](Vcpu::run) again, which re-issues the same read — now
    /// satisfied. No `deliver_*` is needed (unlike the other events, this one carries no pending dst).
    StdinPark,
}

/// A §22 JIT op awaiting the host's [`VcpuEvent::JitInstall`]/`JitUninstall`/`JitInvoke` reply — the
/// vCPU-side residue (dst + the op's parameters) carried across the host round-trip, so the matching
/// `deliver_jit_*` can finish the op against the shared [`Domain`].
enum PendingJit {
    Install {
        dst: u32,
    },
    Uninstall {
        slot: i64,
        dst: u32,
    },
    Invoke {
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        dst: u32,
    },
}

/// One **resumable** vCPU over a shared window. The host calls [`run`](Vcpu::run) to advance it until a
/// [`VcpuEvent`], services the event, delivers the result (`deliver_*`), and runs again — so the same
/// engine semantics work whether the host orchestrates with native threads or wasm Workers. Scope (as
/// for [`drive_parallel`]): `thread.spawn`/`join` + `memory.wait`/`notify` + atomics + compute, §22
/// guest-JIT (`install`/`uninstall`/`invoke`) serviced as host events against the **shared**
/// [`Domain`], and — for a vCPU carrying a powerbox — the §14 domain ops (`spawn_coroutine_module`
/// serviced internally; `instantiate`/`instantiate_module` surfacing [`VcpuEvent::Instantiate`], the
/// child a [`Vcpu::new_confined_child`] on its own Worker). By default carries a deny-all `Host` (an
/// I/O `cap.call` is an inert `CapFault`); attach the run's shared powerbox with
/// [`with_shared_host`](Vcpu::with_shared_host) (THREADS.md 4d) and `cap.call` host I/O works from
/// every vCPU sharing it, serialized per call — `drive_parallel`'s 4c-host model.
pub struct Vcpu<'p> {
    prog: &'p VcpuProgram,
    vt: VTask,
    fibers: Vec<FiberState>,
    fiber_sp: Vec<u64>,
    fiber_meta: Vec<(i32, i64)>,
    mem: Option<Mem>,
    fuel: u64,
    host: Host,
    /// The run's **shared powerbox** (THREADS.md 4d): when set (see
    /// [`with_shared_host`](Vcpu::with_shared_host)), every host access — `cap.call` dispatch, §14
    /// module/authority resolution, an invoked §22 unit's calls — goes through this `Mutex<Host>`
    /// instead of the owned `host`, exactly [`drive_parallel`]'s 4c-host model: each `cap.call` locks
    /// only for its own dispatch, so compute/atomics between calls stay lock-free, and host I/O
    /// (stream writes, clock) works from every vCPU of the run. `None` ⇒ the owned (default deny-all)
    /// host, as before.
    shared_host: Option<&'p std::sync::Mutex<Host>>,
    /// A §14 **confined child**'s own domain (its natural table over the shared source — no parent
    /// §22 install slots); `None` for a root / `thread.spawn` child, which dispatch through
    /// [`VcpuProgram::dom`]'s table (`prog.dom`). The `source` `Arc` is the same either way.
    own_dom: Option<Domain>,
    /// The dst register awaiting a `deliver_*` after a host-serviced event.
    pending: Option<u32>,
    /// A §22 JIT op awaiting its `deliver_jit_*` (carries the op's dst + parameters across the
    /// host round-trip). Distinct from `pending` because the reply payload is richer than one register.
    pending_jit: Option<PendingJit>,
    /// #846 — the **emitted-invoke** fiber registry: while a codegen host services a
    /// [`VcpuEvent::JitInvoke`] on emitted wasm, each cross-tier callback it bounces back here
    /// ([`bounce_call`](Vcpu::bounce_call)) shares this registry, so a fiber parked by one callback
    /// is resumable by a later one — the same one-registry-per-invoke scope the interpreted
    /// `run_invoke` has by construction. Cleared when the invoke resolves (`deliver_jit_invoke_*`).
    invoke_fibers: Vec<FiberState>,
    /// A trap to surface on the next `run` (a joined child trap propagates to the joiner).
    trap: Option<Trap>,
    /// **wasm-JIT tier-up eligibility** (browser wasm-JIT threads slice). When set, `jit_eligible[f]`
    /// means function `f`'s whole reachable region is JIT-compilable and suspension-free, so a direct
    /// `Call` to it is surfaced as a [`VcpuEvent::TierUp`] — the host runs the emitted `f{f}` on the
    /// Worker (top-level caller, so a guest trap is a catchable `RuntimeError`) and delivers the
    /// result back via [`deliver_tierup`](Vcpu::deliver_tierup). `None` ⇒ everything interprets, as
    /// before this seam existed. The engine stays wasm-agnostic: it consults only this bitmap; the
    /// embedder computes it (e.g. from `svm_wasm_jit::analyze`).
    jit_eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 paged tier-up: see the `Vm` field of the same name; mirrored here for the `JitInvoke`
    /// surfacing (which reads the Vcpu, not the Vm).
    jit_page_checked: bool,
    /// A tier-up call awaiting its [`deliver_tierup`](Vcpu::deliver_tierup): the caller-frame-relative
    /// dst slot the emitted region's results land in, and their types (to re-tag the delivered raw
    /// slots — the caller's window base is the one the spill persisted).
    pending_tierup: Option<(usize, Box<[ValType]>)>,
}

impl<'p> Vcpu<'p> {
    /// The **root** vCPU: builds its window over `back` and **seeds + data-initialises** it (the once,
    /// before any child shares it).
    pub fn new_root(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        init_mem: &[u8],
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, Host::new())
    }

    /// Like [`new_root`](Vcpu::new_root), but the vCPU carries a **powerbox** (its own `Host`) instead
    /// of the deny-all default — the seam §14 needs (THREADS.md 4c-domain §14-D). Unlike §22 JIT
    /// (whose ops hand the raw cap handle to the host to resolve), §14 resolves its `Instantiator`
    /// authority **in-Vm** during `resume`, so the grant must live in this vCPU's own host; with it,
    /// `spawn_coroutine_module` is then serviced entirely inside [`run`](Vcpu::run) (no host event).
    /// Grant only the non-I/O caps (`Instantiator`/`Module`) — the resumable path still has no host
    /// I/O, so an I/O `cap.call` remains an inert `CapFault`.
    pub fn new_root_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        init_mem: &[u8],
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// Like [`new_root_with_powerbox`](Vcpu::new_root_with_powerbox), but over an **engine-backed
    /// reservation** (`Mem::with_reservation`) instead of an external `Arc<Region>` — the resumable twin
    /// of [`compile_and_run_capture_reserved_with_host`], which reserves the same way. This is the
    /// persistent-backend seam (the browser Postgres console): a single owned-host vCPU that grows its
    /// heap into the `reserved_log2` tail and stays alive across [`run`](Vcpu::run) parks, so blocking
    /// stdin ([`set_stdin_blocking`](Vcpu::set_stdin_blocking)) can suspend it between queries. Uses the
    /// same `DEFAULT_RESERVED_LOG2`-scale window a one-shot `--single` boot uses; pass that.
    pub fn new_root_reserved_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        init_mem: &[u8],
        host: Host,
        reserved_log2: u8,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation(reserved_log2, sl);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// [`new_root_reserved_with_powerbox`](Self::new_root_reserved_with_powerbox), but the backing
    /// is **caller-provided** (a [`Region::shared`](super::Region) over the host's own window
    /// buffer) rather than engine-owned — the native vehicle for driving wasm-JIT tier-up over a
    /// live, `vm_map`-growable window whose bytes the host must also read (the browser's
    /// shared-linear-memory shape; `tierup_grow_window.rs`). `back` must address the full
    /// `1 << reserved_log2` reservation so grown tail pages land in the caller's buffer.
    pub fn new_root_reserved_over_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        init_mem: &[u8],
        host: Host,
        reserved_log2: u8,
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(reserved_log2, sl, back);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// A `thread.spawn`ed **child** vCPU: shares `back` but does **not** re-seed (the window is already
    /// live with the root's image + every vCPU's writes). Module-0 shorthand for [`new_child_in`].
    pub fn new_child(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        Vcpu::new_child_in(prog, 0, func, args, back)
    }

    /// [`new_child`], but `func` resolves in `module` of the shared source and the child's root frame
    /// starts there — the constructor for a spawn issued by an **installed §22 unit's** code
    /// ([`VcpuEvent::Spawn`] carries the spawning frame's module; CONSOLIDATION.md §11). The child
    /// keeps `own_dom: None`: a thread shares its spawner's dispatch table, whichever module its
    /// frames start in.
    pub fn new_child_in(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        let sl = prog.mem_size_log2;
        Self::new_child_with(prog, module, func, args, back, sl)
    }

    /// [`new_child_in`] with the window mask chosen by the caller: a thread shares its **spawner's**
    /// window, which for a §14 confined spawner is its carve — smaller than the guest module's
    /// declared memory — so the driver passes the actual window's `size_log2` (CONSOLIDATION.md §11).
    pub fn new_child_sized(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        size_log2: u8,
    ) -> Result<Vcpu<'p>, Trap> {
        Self::new_child_with(prog, module, func, args, back, Some(size_log2))
    }

    fn new_child_with(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        size_log2: Option<u8>,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = size_log2.map(|sl| Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back));
        Vcpu::with_mem_in(prog, module, func, args, mem, Host::new())
    }

    fn with_mem(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        mem: Option<Mem>,
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        Vcpu::with_mem_in(prog, 0, func, args, mem, host)
    }

    fn with_mem_in(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        mem: Option<Mem>,
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        let cm = prog
            .dom
            .source
            .get(module as usize)
            .ok_or(Trap::Malformed)?;
        if func as usize >= cm.progs.len() {
            return Err(Trap::Malformed);
        }
        let mut vt = VTask::new(&cm, func as usize, args)?;
        vt.active.module = module as usize;
        vt.active.home = module as usize;
        Ok(Vcpu {
            vt,
            fibers: Vec::new(),
            fiber_sp: Vec::new(),
            fiber_meta: Vec::new(),
            mem,
            fuel: u64::MAX,
            host,
            shared_host: None,
            own_dom: None,
            prog,
            pending: None,
            pending_jit: None,
            invoke_fibers: Vec::new(),
            trap: None,
            jit_eligible: None,
            jit_page_checked: false,
            pending_tierup: None,
        })
    }

    /// A §14 **confined executor child** vCPU (THREADS.md 4c-domain §14-D2) — what the host starts on
    /// its own Worker/thread in response to [`VcpuEvent::Instantiate`]. `back` must be a region over
    /// exactly the parent's carve (`len == 1 << size_log2`): per DESIGN.md §14, a sub-window is
    /// indistinguishable from a top-level window, so the carve region simply *is* the child's window
    /// (its bytes — anything the parent wrote there, an op-5 child's materialized data segments — are
    /// already in the shared memory; nothing is re-seeded). Builds internally:
    ///   * the **attenuated powerbox** — an `Instantiator` and an `AddressSpace`, each over the
    ///     child's own `[0, 2^size_log2)`, passed as the entry args (one or both, per the entry's
    ///     signature) — so the child can itself nest, and no authority ever crosses the host;
    ///   * the child's **own domain** — a natural table over `module` in the shared source (no parent
    ///     §22 install slots — the fresh table is the confinement).
    ///
    /// `module`/`entry`/`size_log2`/`fuel` come verbatim from the event.
    pub fn new_confined_child(
        prog: &'p VcpuProgram,
        module: u32,
        entry: u32,
        back: std::sync::Arc<super::Region>,
        size_log2: u8,
        fuel: u64,
    ) -> Result<Vcpu<'p>, Trap> {
        // The plain op-0 confined child: attenuated `Instantiator`+`AddressSpace` only, no re-granted
        // I/O caps (the browser/native drivers' path). Delegates with a no-op grant installer.
        Self::new_confined_child_granted(prog, module, entry, back, size_log2, fuel, &mut |_| {})
    }

    /// Like [`new_confined_child`](Self::new_confined_child), but the caller may install **re-granted
    /// caps** into the child's powerbox before it runs — the §14 op-13 grant list (a shared `fs`, an
    /// inherited `stdout`), reached by the child through `cap.self.resolve` (#1011 slice 3a). The
    /// `install_grants` closure receives the freshly-built child `Host` (with its `Instantiator`+
    /// `AddressSpace` already granted) and, for each grant, calls the parent's
    /// [`Host::regrant_into_child`] + [`Host::register_cap_name`] — so the grant *policy* (which handles
    /// are `can_regrant`-eligible) stays with the caller (the interpreter's cap dispatch), and this
    /// constructor stays pure mechanism (INVARIANTS §4). A confined child so built still masks every
    /// window access to its own carve (§2 unchanged) — a re-granted cap is a cross-tier `cap.call`, not
    /// a window access.
    pub fn new_confined_child_granted(
        prog: &'p VcpuProgram,
        module: u32,
        entry: u32,
        back: std::sync::Arc<super::Region>,
        size_log2: u8,
        fuel: u64,
        install_grants: &mut dyn FnMut(&mut Host),
    ) -> Result<Vcpu<'p>, Trap> {
        if size_log2 >= 64 {
            return Err(Trap::Malformed);
        }
        let cunit = prog
            .dom
            .source
            .get(module as usize)
            .ok_or(Trap::Malformed)?;
        // One or two entry args, per the signature the parent already validated (its starter caps).
        let want_as = cunit
            .sigs
            .get(entry as usize)
            .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
        let child_size = 1u64 << size_log2;
        let mut host = Host::new();
        let cinst = host.grant_instantiator(0, child_size);
        let cas = host.grant_address_space(0, child_size);
        // Install any re-granted caps (op-13 grant list) into the child powerbox under their names. The
        // starter entry args stay `[Instantiator, AddressSpace]`; re-granted caps are name-resolved.
        install_grants(&mut host);
        let args = if want_as {
            vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
        } else {
            vec![Value::I64(cinst as i64)]
        };
        let mem = Some(Mem::with_reservation_over(
            DEFAULT_RESERVED_LOG2,
            size_log2,
            back,
        ));
        let mut vt = VTask::new(&cunit, entry as usize, &args)?;
        vt.active.module = module as usize;
        vt.active.home = module as usize;
        let own_dom = Domain::child(
            std::sync::Arc::clone(&prog.dom.source),
            build_table_for(cunit.progs.len(), 0, module),
        );
        Ok(Vcpu {
            vt,
            fibers: Vec::new(),
            fiber_sp: Vec::new(),
            fiber_meta: Vec::new(),
            mem,
            fuel,
            host,
            shared_host: None,
            own_dom: Some(own_dom),
            prog,
            pending: None,
            pending_jit: None,
            invoke_fibers: Vec::new(),
            trap: None,
            jit_eligible: None,
            jit_page_checked: false,
            pending_tierup: None,
        })
    }

    /// Attach the run's **shared powerbox** (THREADS.md 4d — builder-style, on any constructor's
    /// result): every host access of this vCPU then goes through `host` under its lock, so `cap.call`
    /// (host I/O), §14 module/authority resolution, and invoked §22 units work from every vCPU of the
    /// run sharing it — the resumable counterpart of [`drive_parallel`]'s 4c-host shared `Mutex<Host>`.
    /// The embedder grants into the `Host` *before* the run (handle order is deterministic) and reads
    /// its state (e.g. `stdout`) after; per-call serialization is the documented 4c-host model.
    pub fn with_shared_host(mut self, host: &'p std::sync::Mutex<Host>) -> Vcpu<'p> {
        self.shared_host = Some(host);
        self
    }

    /// Attach the **wasm-JIT tier-up bitmap** (browser wasm-JIT threads slice, builder-style). A
    /// direct `Call` to a function `f` with `eligible[f] == true` then surfaces as
    /// [`VcpuEvent::TierUp`] instead of interpreting `f` — the host runs the emitted region and
    /// `deliver_tierup`s the result. `eligible.len()` should cover the primary module's functions;
    /// an out-of-range index is treated as not-eligible (interprets).
    pub fn with_jit_eligible(mut self, eligible: std::sync::Arc<[bool]>) -> Vcpu<'p> {
        self.vt.active.jit_eligible = Some(std::sync::Arc::clone(&eligible));
        self.jit_eligible = Some(eligible);
        self
    }

    /// #750 **paged tier-up**: mark the eligible set as emitted with the software page-check
    /// (`compile_module_tierup_paged`). The dispatch then surfaces tier-up regardless of the
    /// window's scalar representability — the event's `mapped` is the reserved window size, and the
    /// driver's page-state table (refreshed per call from [`mem_map_info`](Vcpu::mem_map_info))
    /// carries the per-page fidelity the scalar cannot.
    pub fn with_jit_page_checked(mut self) -> Vcpu<'p> {
        self.vt.active.jit_page_checked = true;
        self.jit_page_checked = true;
        self
    }

    /// The run's window memory-map introspection ([`MemMapInfo`]) — what a #750 page-checked
    /// driver rebuilds its byte-per-page state table from before each emitted call (page state is
    /// frozen while emitted code runs: page ops are `cap.call`s, which are never emitted and never
    /// reachable through a cross-tier leaf). `None` for a memory-less module.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// #1009 paged tier-up: the window's page-map version — a cheap `O(1)` counter bumped on every
    /// `map`/`unmap`/`protect`. A page-checked driver caches the page-state table it built from
    /// [`mem_map_info`](Vcpu::mem_map_info) and rebuilds only when this changes (the table is
    /// identical between two tier-ups with no intervening page-op). `0` for a memory-less module.
    pub fn mem_map_version(&self) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.map_version())
    }

    /// Reclaim this vCPU's live guest window after it finishes — the seam a **reactor** uses to keep
    /// the window (globals, BSS, and the `vm_map`-grown heap, with its address-space commit state)
    /// alive across per-frame vCPUs: build a vCPU over the persistent [`Mem`] with
    /// [`with_mem`](Vcpu::with_mem), run one frame to `Done`, then `take_mem` it back for the next
    /// frame. `None` for a memory-less module (or if already taken).
    pub(crate) fn take_mem(&mut self) -> Option<Mem> {
        self.mem.take()
    }

    /// Enable **blocking stdin** on this vCPU's owned powerbox (a persistent interactive session — the
    /// browser Postgres console). A `read` on an exhausted stdin buffer then surfaces
    /// [`VcpuEvent::StdinPark`] instead of returning EOF; feed more input with [`push_stdin`](Vcpu::push_stdin)
    /// and call [`run`](Vcpu::run) again. Only meaningful for an owned-host vCPU (not `with_shared_host`).
    pub fn set_stdin_blocking(&mut self, on: bool) {
        self.host.set_stdin_blocking(on);
    }

    /// Append bytes to this vCPU's stdin buffer, then [`run`](Vcpu::run) again to satisfy a pending
    /// [`VcpuEvent::StdinPark`] (or to preload input before the first `run`).
    pub fn push_stdin(&mut self, bytes: &[u8]) {
        self.host.push_stdin(bytes);
    }

    /// Borrow this vCPU's owned powerbox — e.g. to read `stdout` after a [`run`](Vcpu::run) that parked
    /// or finished. `None`-safe only for an owned host; a `with_shared_host` vCPU services I/O through
    /// the shared lock, not here.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Advance this vCPU until it finishes, traps, or hits a host-serviced event. The host must
    /// `deliver_*` the result of any `Spawn`/`Join`/`Wait`/`Notify` before calling `run` again.
    pub fn run(&mut self) -> VcpuEvent {
        if let Some(t) = self.trap.take() {
            return VcpuEvent::Trapped(t);
        }
        debug_assert!(
            self.pending.is_none(),
            "deliver the last event before resuming"
        );
        // Loop so §14 `spawn_coroutine_module` (serviced in-Rust against this vCPU's own powerbox)
        // never surfaces to the orchestrating host — it only ever sees the multi-vCPU events
        // `spawn`/`join`/`wait`/`notify`, the §22 JIT events, and §14 `Instantiate` (+ `done`/`trap`).
        loop {
            // A §14 confined child dispatches through its OWN domain (own natural table, no parent
            // install slots); everything else through the program's shared one. Host access goes
            // through the run's shared powerbox when attached (4d), else the owned host.
            let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
            let mut ctx = RunCtx {
                table: &dom.table,
                fuel: &mut self.fuel,
                mem: &mut self.mem,
                durable: false,
                host: match self.shared_host {
                    Some(m) => HostCell::Shared(m),
                    None => HostCell::Excl(&mut self.host),
                },
            };
            let stop = step_vcpu(
                &mut self.vt,
                &mut self.fibers,
                &mut self.fiber_sp,
                &mut self.fiber_meta,
                dom,
                &mut ctx,
                u64::MAX,
                false, // single-vCPU `Vcpu::run`: no cooperative waker topology (I48 idle N/A)
            );
            match stop {
                // §3.6 (I36 slice 2): live calls / svc.wait / child_offer need the cooperative
                // scheduler's waker topology (`drive`); on this single-vCPU driver nothing could
                // ever wake or mint them — fail closed rather than hang. Unreachable through the
                // compile (op-14 implies the drive path); requires a hand-wired live cap.
                Ok(VcpuStop::LiveCall { .. })
                | Ok(VcpuStop::SvcWait)
                | Ok(VcpuStop::ChildOffer { .. })
                | Ok(VcpuStop::CloneCaller { .. })
                | Ok(VcpuStop::Reap { .. })
                // I48: `BlockOnFiber` is a cooperative-driver idle (this path passes
                // `cooperative: false`, so it never arises here); fail closed like its neighbours.
                | Ok(VcpuStop::BlockOnFiber { .. }) => return VcpuEvent::Trapped(Trap::ThreadFault),
                Err(t) => return VcpuEvent::Trapped(t),
                Ok(VcpuStop::Done(vals)) => return VcpuEvent::Done(vals),
                Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                }) => {
                    self.pending_tierup = Some((dst, results));
                    return VcpuEvent::TierUp { func, argv, mapped };
                }
                Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module,
                }) => {
                    // Bound-check `func` in the SPAWNING FRAME's module (an installed §22 unit spawns
                    // its own functions — CONSOLIDATION.md §11), not module 0.
                    let ok = dom
                        .source
                        .get(module as usize)
                        .is_some_and(|c| (func as usize) < c.progs.len());
                    if !ok {
                        return VcpuEvent::Trapped(Trap::Malformed);
                    }
                    self.pending = Some(dst);
                    return VcpuEvent::Spawn {
                        func,
                        sp,
                        arg,
                        module,
                    };
                }
                Ok(VcpuStop::Join { handle, dst }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Join { handle };
                }
                Ok(VcpuStop::CapPending { id, dst }) => {
                    // F2: the session driver keeps the inline completion wait — identical to
                    // the pre-F2 in-op wait (the I45 whole-vCPU posture for this driver).
                    let comps = match self.shared_host {
                        Some(m) => m.lock_unpoisoned().completions(),
                        None => self.host.completions(),
                    };
                    let r = comps.wait(id);
                    self.vt.active.set(dst, Reg::from_i64(r));
                }
                Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Wait {
                        addr: base,
                        expected,
                        width,
                        timeout,
                    };
                }
                Ok(VcpuStop::Notify { base, count, dst }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Notify { addr: base, count };
                }
                // §22 guest-JIT — the host resolves the unit (it holds the powerbox), the vCPU
                // installs / invokes it against the **shared** [`Domain`]. The op's residue is parked
                // in `pending_jit` until the matching `deliver_jit_*`.
                Ok(VcpuStop::JitInstall { h, code, dst }) => {
                    self.pending_jit = Some(PendingJit::Install { dst });
                    return VcpuEvent::JitInstall { handle: h, code };
                }
                Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                    self.pending_jit = Some(PendingJit::Uninstall { slot, dst });
                    return VcpuEvent::JitUninstall { handle: h, slot };
                }
                Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                }) => {
                    self.pending_jit = Some(PendingJit::Invoke {
                        argv: argv.clone(),
                        params: params.clone(),
                        results: results.clone(),
                        dst,
                    });
                    // #717 host sync: snapshot the window's scalar committed extent for a codegen
                    // host — `Some(H)` goes into the emitted unit's `"mapped"` global; `None`
                    // (unrepresentable page state) tells it to decline emitted execution and use
                    // the interpreted delivery instead. A memory-less module has nothing to bound.
                    // #750: a page-checked run surfaces the reserved size instead (see the tier-up
                    // dispatch), the table carrying per-page fidelity.
                    let mapped = match self.mem.as_ref() {
                        None => Some(0),
                        Some(m) if self.jit_page_checked => Some(m.reserved_size()),
                        Some(m) => m.scalar_extent(),
                    };
                    return VcpuEvent::JitInvoke {
                        handle: h,
                        code,
                        argv,
                        params,
                        results,
                        mapped,
                    };
                }
                // §14 executor children (THREADS.md 4c-domain §14-D2): this vCPU does all the
                // authority-bearing validation/preparation, then surfaces a mechanical
                // [`VcpuEvent::Instantiate`] for the host (a bad carve/entry lands `-EINVAL` in place
                // and the run continues; a bad module handle traps).
                Ok(VcpuStop::Instantiate {
                    ibase,
                    isize: isz,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                }) => {
                    // op 11 (named grants) is driven by the scheduler `drive` arm (the browser's
                    // `compile_and_run_with_host` path); this standalone single-vCPU resume path builds
                    // no child powerbox, so it declines a grant list rather than silently drop it.
                    if grants.is_some() {
                        return VcpuEvent::Trapped(Trap::Malformed);
                    }
                    match self
                        .event_instantiate(ibase, isz, entry, off, size_log2, quota, budget, dst)
                    {
                        Ok(Some(ev)) => return ev,
                        Ok(None) => {} // -EINVAL landed in place — keep running
                        Err(t) => return VcpuEvent::Trapped(t),
                    }
                }
                Ok(VcpuStop::InstantiateModule {
                    ibase,
                    isize: isz,
                    mh,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                }) => {
                    // op 13 (named grants) is driven by the scheduler `drive` arm (the browser's
                    // `compile_and_run_with_host` path); this standalone single-vCPU resume path builds
                    // no child powerbox, so it declines a grant list rather than silently drop it.
                    if grants.is_some() {
                        return VcpuEvent::Trapped(Trap::Malformed);
                    }
                    match self.event_instantiate_module(
                        ibase, isz, mh, entry, off, size_log2, quota, budget, dst,
                    ) {
                        Ok(Some(ev)) => return ev,
                        Ok(None) => {} // -EINVAL landed in place — keep running
                        Err(t) => return VcpuEvent::Trapped(t),
                    }
                }
                // Blocking-stdin park: the guest read an exhausted stdin under `set_stdin_blocking`.
                // Nothing to deliver — `pc` was left at the read, so pushing input + `run()` again
                // re-issues it. Surface to the host, which pumps the session.
                Ok(VcpuStop::StdinPark) => return VcpuEvent::StdinPark,
            }
        }
    }

    /// Validate + prepare a §14 `instantiate` (op 0 — or a §3d op-17 record with no module) and
    /// produce its [`VcpuEvent::Instantiate`], or land `-EINVAL` in place (`Ok(None)`) on a bad
    /// entry/carve — identical checks to the cooperative and parallel drivers' arms. A record's
    /// `budget` (`0` = none) is funded here — the commit site — via [`take_spawn_budget`]
    /// (peek-then-drain: every `-EINVAL` above leaves it intact); a handle that vanished since the
    /// exec arm's peek (a shared-powerbox race) is the one `Err` (`CapFault`).
    #[allow(clippy::too_many_arguments)]
    fn event_instantiate(
        &mut self,
        ibase: u64,
        isize: u64,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        budget: i32,
        dst: u32,
    ) -> Result<Option<VcpuEvent>, Trap> {
        // The child runs in the CALLING frame's module — module 0 for a root/plain guest, the granted
        // module for an `instantiate_module` child whose entry itself instantiates (§14 nesting
        // composes across `instantiate_module`). Validating against `primary()` here used to send a
        // module-child's nested instantiate to `-EINVAL` (its entry index doesn't exist in module 0).
        let cur_module = self.vt.active.module;
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let Some(cm) = dom.source.get(cur_module) else {
            self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            return Ok(None);
        };
        let ok_entry = cm
            .sigs
            .get(entry as usize)
            .is_some_and(|(p, r)| child_entry_ok(p, r));
        let child_size = if (0..64).contains(&size_log2) {
            1u64 << size_log2
        } else {
            0
        };
        let off_u = off as u64;
        let fits = carve_fits(
            off_u,
            size_log2,
            isize,
            ibase,
            self.mem.as_ref().map_or(0, |m| m.null_guard),
        );
        if !ok_entry || !fits {
            self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            return Ok(None);
        }
        // Window-relative carve (`window.base()` is 0 for this path's top-level region windows; the
        // term keeps exact parity with the drive/parallel arms' backing-absolute math).
        let pbase = self.mem.as_ref().map_or(0, |m| m.window.base());
        let carve = pbase + ibase + off_u;
        let fuel = if budget != 0 {
            let pf = self.fuel;
            let take = match self.shared_host {
                Some(m) => {
                    let mut g = m.lock_unpoisoned();
                    take_spawn_budget(&mut g, budget, child_size, pf)?
                }
                None => take_spawn_budget(&mut self.host, budget, child_size, pf)?,
            };
            match take {
                Some(f) => f,
                None => {
                    self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    return Ok(None);
                }
            }
        } else if quota <= 0 {
            self.fuel
        } else {
            (quota as u64).min(self.fuel)
        };
        self.pending = Some(dst);
        Ok(Some(VcpuEvent::Instantiate {
            module: cur_module as u32,
            entry: entry as u32,
            carve,
            size_log2: size_log2 as u8,
            fuel,
        }))
    }

    /// Validate + prepare a §14 `instantiate_module` (op 5, separate-module child) and produce its
    /// [`VcpuEvent::Instantiate`]: resolve the granted `Module` from this vCPU's own powerbox
    /// (`Err` — a forged/closed handle — traps), compile it, **push it to the shared source**, and
    /// materialize its data segments into the carve *before* the event surfaces (the spawn hand-off
    /// is the happens-before, so the child Worker observes them). `Ok(None)` lands `-EINVAL` in place
    /// on a bad entry/carve/memory mismatch.
    #[allow(clippy::too_many_arguments)]
    fn event_instantiate_module(
        &mut self,
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        budget: i32,
        dst: u32,
    ) -> Result<Option<VcpuEvent>, Trap> {
        // Resolve the granted module from the run's powerbox (the shared one when attached).
        let (cfuncs, cmem_log2, cdata) = match self.shared_host {
            Some(m) => {
                let g = m.lock_unpoisoned();
                let g = g.resolve_module(mh)?;
                (g.funcs.clone(), g.memory_log2, g.data.clone())
            }
            None => {
                let g = self.host.resolve_module(mh)?;
                (g.funcs.clone(), g.memory_log2, g.data.clone())
            }
        };
        let child_compiled = compile_module(&cfuncs).ok_or(Trap::Malformed)?;
        let ok_entry = child_compiled
            .sigs
            .get(entry as usize)
            .is_some_and(|(p, r)| child_entry_ok(p, r));
        let child_size = if (0..64).contains(&size_log2) {
            1u64 << size_log2
        } else {
            0
        };
        let off_u = off as u64;
        let fits = carve_fits(
            off_u,
            size_log2,
            isize,
            ibase,
            self.mem.as_ref().map_or(0, |m| m.null_guard),
        );
        let mod_ok = cmem_log2 == Some(size_log2 as u8);
        if !ok_entry || !fits || !mod_ok {
            self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            return Ok(None);
        }
        let pbase = self.mem.as_ref().map_or(0, |m| m.window.base());
        let carve = pbase + ibase + off_u;
        if let Some(m) = self.mem.as_ref() {
            for d in cdata.iter() {
                if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
                    for (k, &b) in d.bytes.iter().enumerate() {
                        m.set_byte(carve + d.offset + k as u64, b);
                    }
                }
            }
        }
        let cm = self.prog.dom.source.push(child_compiled);
        // A record's budget funds the child at this commit site (see `event_instantiate`).
        let fuel = if budget != 0 {
            let pf = self.fuel;
            let take = match self.shared_host {
                Some(m) => {
                    let mut g = m.lock_unpoisoned();
                    take_spawn_budget(&mut g, budget, child_size, pf)?
                }
                None => take_spawn_budget(&mut self.host, budget, child_size, pf)?,
            };
            match take {
                Some(f) => f,
                None => {
                    self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    return Ok(None);
                }
            }
        } else if quota <= 0 {
            self.fuel
        } else {
            (quota as u64).min(self.fuel)
        };
        self.pending = Some(dst);
        Ok(Some(VcpuEvent::Instantiate {
            module: cm as u32,
            entry: entry as u32,
            carve,
            size_log2: size_log2 as u8,
            fuel,
        }))
    }

    /// Deliver a `thread.spawn` handle (after `Spawn`).
    pub fn deliver_handle(&mut self, handle: i32) {
        self.deliver_code(handle);
    }

    /// Deliver a `Wait` wasm code or a `Notify` woken-count into the pending dst.
    pub fn deliver_code(&mut self, v: i32) {
        let dst = self.pending.take().expect("deliver with no pending event");
        self.vt.active.set(dst, Reg::from_i32(v));
    }

    /// Deliver a joined child's result (after `Join`): its first value lands in the joiner's dst, or a
    /// child trap propagates (the joiner traps on its next `run`).
    pub fn deliver_join(&mut self, res: Result<Vec<Value>, Trap>) {
        let dst = self.pending.take().expect("deliver with no pending event");
        match res {
            Ok(vals) => {
                let v = vals.first().copied().unwrap_or(Value::I64(0));
                self.vt.active.set(dst, Reg::from_value(v));
            }
            Err(t) => self.trap = Some(t),
        }
    }

    /// Deliver the resolved unit funcs for a `JitInstall` (the host resolved authority + code-handle):
    /// `Err` (forged / cross-domain / wrong-type handle) propagates as a trap; `Ok(funcs)` is compiled
    /// and installed into the **shared** [`Domain`] (so every vCPU/Worker can `call_indirect` it), the
    /// slot — or `-ENOSPC` if the table is full / `Malformed` if the unit is outside engine coverage —
    /// written to the awaiting dst.
    ///
    /// Returns `Some(slot)` iff the unit was actually installed (the slot the guest received), else
    /// `None` (trap / `-ENOSPC`). A wasm-tier host uses this to mirror the shared `Domain` slot into a
    /// per-Worker `WebAssembly.Table` (§22 Model B2 cross-Worker) — funcrefs can't cross Workers, so
    /// each Worker learns *which slot* an install filled and populates its own table. The `Domain`
    /// itself stays wasm-agnostic; the slot→code-handle→emitted-wasm mapping lives in the host.
    pub fn deliver_jit_install(
        &mut self,
        funcs: Result<std::sync::Arc<[Func]>, Trap>,
    ) -> Option<usize> {
        let Some(PendingJit::Install { dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_install with no pending install");
        };
        let funcs = match funcs {
            Ok(f) => f,
            Err(t) => {
                self.trap = Some(t);
                return None;
            }
        };
        let (res, slot) = match compile_module(&funcs) {
            // Install into THIS vCPU's domain (== the shared one for a root; a §14 confined child —
            // which can't hold a Jit cap anyway — would only ever fill its own table).
            Some(unit) => match self
                .own_dom
                .as_ref()
                .unwrap_or(&self.prog.dom)
                .install(unit)
            {
                Some(slot) => (slot as i64, Some(slot)),
                None => (super::ENOSPC, None),
            },
            None => {
                self.trap = Some(Trap::Malformed); // unit op outside coverage
                return None;
            }
        };
        self.vt.active.set(dst, Reg::from_i64(res));
        slot
    }

    /// Deliver the authority check for a `JitUninstall`: `Err` propagates as a trap; `Ok(())` clears the
    /// shared table `slot` (`0` on success, `EINVAL` for a real-func / out-of-range / already-empty slot).
    ///
    /// Returns `Some(slot)` iff a slot was actually cleared, so a wasm-tier host can null the matching
    /// per-Worker `WebAssembly.Table` slot (the `deliver_jit_install` counterpart) — keeping each
    /// Worker's mirror exact so a stale `call_indirect` traps.
    pub fn deliver_jit_uninstall(&mut self, authorized: Result<(), Trap>) -> Option<usize> {
        let Some(PendingJit::Uninstall { slot, dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_uninstall with no pending uninstall");
        };
        if let Err(t) = authorized {
            self.trap = Some(t);
            return None;
        }
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let n_real = dom.source.primary().progs.len();
        let cleared = dom.uninstall(slot as usize, n_real);
        self.vt
            .active
            .set(dst, Reg::from_i64(if cleared { 0 } else { super::EINVAL }));
        cleared.then_some(slot as usize)
    }

    /// Deliver the resolved unit funcs for a `JitInvoke`: `Err` propagates as a trap; `Ok(funcs)` is
    /// compiled, arity-checked against the call signature (`CapFault` on mismatch), then run
    /// synchronously over this vCPU's window — its results marshalled to the awaiting dst. The invoked
    /// unit runs over this vCPU's (deny-all) powerbox, so a unit that itself makes a `cap.call` faults;
    /// a powerbox-backed unit is the orchestrator's responsibility (see [`Vcpu`]).
    pub fn deliver_jit_invoke(&mut self, funcs: Result<std::sync::Arc<[Func]>, Trap>) {
        let Some(PendingJit::Invoke {
            argv,
            params,
            results,
            dst,
        }) = self.pending_jit.take()
        else {
            panic!("deliver_jit_invoke with no pending invoke");
        };
        let funcs = match funcs {
            Ok(f) => f,
            Err(t) => {
                self.trap = Some(t);
                return;
            }
        };
        let unit = match compile_module(&funcs) {
            Some(u) => u,
            None => {
                self.trap = Some(Trap::Malformed);
                return;
            }
        };
        let arity_ok = unit
            .sigs
            .first()
            .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
        if !arity_ok {
            self.trap = Some(Trap::CapFault);
            return;
        }
        let child_args: Vec<Value> = params
            .iter()
            .zip(argv.iter())
            .map(|(ty, s)| slot_to_val(*ty, *s))
            .collect();
        // The effective domain borrows only `self.own_dom`/`self.prog` (shared) — disjoint from the
        // `&mut self.fuel/mem/host` fields the invoke needs, so the borrows split.
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let umod = dom.source.push(unit);
        // The invoked unit runs over the run's powerbox — the shared one when attached (its
        // `cap.call`s then serialize per-call like every other vCPU's, matching `drive_parallel`),
        // else this vCPU's owned (default deny-all) host.
        let mut cell = match self.shared_host {
            Some(m) => HostCell::Shared(m),
            None => HostCell::Excl(&mut self.host),
        };
        match run_invoke(
            &dom.source,
            &dom.table,
            umod,
            &child_args,
            &mut self.fuel,
            &mut self.mem,
            &mut cell,
        ) {
            Ok(vals) => {
                for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                    let re = slot_to_val(*ty, val_to_slot(*v));
                    self.vt.active.set(dst + i as u32, Reg::from_value(re));
                }
            }
            Err(t) => self.trap = Some(t),
        }
    }

    /// Deliver the **results** of a [`VcpuEvent::JitInvoke`] the host ran on **emitted wasm** (the
    /// browser's real-codegen §22 tier) instead of the engine interpreting the unit. Writes the raw
    /// i64 result slots into the awaiting `dst` and resumes — the invoke then looks exactly like the
    /// interpreted [`deliver_jit_invoke`](Vcpu::deliver_jit_invoke) that ran the unit itself. This is
    /// the alternative to that method: a host that emits wasm for the unit (`f{entry}(win, env,
    /// args)`) calls this with the emitted region's results; a host that interprets calls the other.
    /// Too few results is a `Malformed` trap (a mis-marshalled host reply).
    pub fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        self.invoke_fibers.clear(); // the emitted invoke resolved — its bounce registry dies with it
        let Some(PendingJit::Invoke { results, dst, .. }) = self.pending_jit.take() else {
            panic!("deliver_jit_invoke_vals with no pending invoke");
        };
        if vals.len() < results.len() {
            self.trap = Some(Trap::Malformed);
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.vt
                .active
                .set(dst + i as u32, Reg::from_value(slot_to_val(*ty, vals[i])));
        }
    }

    /// #846 slice 1 — service **one cross-tier bounce** out of an emitted §22 unit: the codegen host
    /// is mid-way through running a [`VcpuEvent::JitInvoke`] on emitted wasm (this vCPU is parked on
    /// the pending invoke), and the unit reached a call the emit routed to `env.call_interp` — a
    /// trampoline'd `call_indirect` target or a cross-tier direct call. `target` resolves through
    /// the **shared dispatch table** exactly as [`Op::CallIndirect`] does (the natural prefix maps a
    /// program function's index to itself; an installed unit sits at its install slot; empty padding
    /// is an `IndirectCallType` trap), and the resolved function runs on a nested interpretation
    /// over this vCPU's **live** window/powerbox/fuel — observably identical to the same call inside
    /// an interpreted invoke. Fibers are serviced against the persistent emitted-invoke registry
    /// ([`invoke_fibers`](Vcpu::invoke_fibers)), so a fiber parked by one callback is resumable by a
    /// later one within the same invoke.
    ///
    /// `io` carries the i64 arg slots in and the result slots out (the `env.call_interp` scratch
    /// ABI — floats by bits, i32s in the low half). Returns the result count. `Err` is the
    /// callback's trap — the host must unwind the emitted unit and deliver it via
    /// [`deliver_jit_invoke_trap`](Vcpu::deliver_jit_invoke_trap) (an `Exit` included: it must
    /// resolve the invoke as the interpreted path would, not be swallowed).
    pub fn bounce_call(&mut self, target: u32, io: &mut [i64]) -> Result<usize, Trap> {
        step(&mut self.fuel, None)?; // fuel unification: the dispatch-site safepoint
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let slot = (target as usize) & (dom.table.len() - 1);
        let ts = dom.table.slot(slot);
        if ts.module == super::TABLE_EMPTY {
            return Err(Trap::IndirectCallType);
        }
        let tm = dom.source.get(ts.module as usize).ok_or(Trap::Malformed)?;
        let (cp, cr) = tm.sigs[ts.func as usize].clone();
        if cp.len() > io.len() || cr.len() > io.len() {
            return Err(Trap::Malformed); // scratch too small — a mis-marshalled host call
        }
        // The i64-slot transport carries scalars only; a v128-sig target can never have been given
        // a trampoline (the host gates that at open) — a bounce naming one is a mis-wired host.
        let scalar =
            |t: &ValType| matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64);
        if !cp.iter().all(scalar) || !cr.iter().all(scalar) {
            return Err(Trap::Malformed);
        }
        let args: Vec<Value> = cp
            .iter()
            .zip(io.iter())
            .map(|(ty, s)| slot_to_val(*ty, *s))
            .collect();
        let mut vm = Vm::new(&tm, ts.func as usize, &args)?;
        vm.module = ts.module as usize;
        let mut cell = match self.shared_host {
            Some(m) => HostCell::Shared(m),
            None => HostCell::Excl(&mut self.host),
        };
        // Registry context (#880): during an emitted **invoke**, callbacks share the
        // invoke-confined registry (`run_invoke` parity — fibers die with the invoke). During an
        // emitted **TIERUP region**, the callback is interpreted-inline-call territory: its fibers
        // register in the vCPU's *run-level* registry (parallel arrays mirrored), so one created
        // here persists for the run to resume later — exactly as the same call inline would.
        let vals = if self.pending_jit.is_some() {
            drive_nested(
                &dom.source,
                &dom.table,
                vm,
                &mut self.fuel,
                &mut self.mem,
                &mut cell,
                &mut self.invoke_fibers,
                None,
            )?
        } else {
            drive_nested(
                &dom.source,
                &dom.table,
                vm,
                &mut self.fuel,
                &mut self.mem,
                &mut cell,
                &mut self.fibers,
                Some((&mut self.fiber_sp, &mut self.fiber_meta)),
            )?
        };
        for (i, v) in vals.iter().enumerate() {
            io[i] = val_to_slot(*v);
        }
        Ok(cr.len())
    }

    /// The window's committed **scalar extent** right now ([`Mem::scalar_extent`]) — the #717 value
    /// the codegen host re-syncs to every live instance's `"mapped"` global after a
    /// [`bounce_call`](Vcpu::bounce_call) (a bounced callback may have `vm_map`-grown the window
    /// mid-invoke; the fan-out makes the growth visible to the emitted tier exactly when the
    /// interpreted path would see it — after the call returns). `0` when the window state is no
    /// longer scalar-representable — deny-everything, the diverge-toward-refusal posture
    /// (INVARIANTS.md #9), or when there is no window.
    pub fn window_scalar_extent(&self) -> u64 {
        self.mem
            .as_ref()
            .and_then(|m| m.scalar_extent())
            .unwrap_or(0)
    }

    /// Deliver a **trap** from a host-run [`VcpuEvent::JitInvoke`] unit (the emitted region hit a
    /// guest `unreachable` / memory fault / div-by-zero / out-of-fuel, surfaced to the host as a
    /// catchable `RuntimeError`). The vCPU traps on its next `run`, exactly as an interpreted invoke
    /// trap would (`deliver_jit_invoke` sets `self.trap` on the unit's `Err`).
    pub fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        self.invoke_fibers.clear(); // the emitted invoke resolved — its bounce registry dies with it
        self.pending_jit = None;
        self.trap = Some(trap);
    }

    /// Deliver the results of a [`VcpuEvent::TierUp`]: the emitted region returned `vals` (raw i64
    /// result slots, one per the callee's result type). Re-tag each into the awaiting `dst` slot(s) of
    /// the caller's window and resume — the tier-up call then looks exactly like an interpreted call
    /// that returned. Too few results is a `Malformed` trap (a mis-marshalled host reply).
    pub fn deliver_tierup(&mut self, vals: &[i64]) {
        let Some((dst, results)) = self.pending_tierup.take() else {
            panic!("deliver_tierup with no pending tier-up");
        };
        if vals.len() < results.len() {
            self.trap = Some(Trap::Malformed);
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.vt.active.set(
                dst as u32 + i as u32,
                Reg::from_value(slot_to_val(*ty, vals[i])),
            );
        }
    }

    /// Deliver a **trap** from a [`VcpuEvent::TierUp`] region (the emitted `f{func}` hit a guest
    /// `unreachable` / memory fault / div-by-zero / out-of-fuel, surfaced to the host as a catchable
    /// `RuntimeError`). The vCPU traps on its next `run`, exactly as if the interpreted call had.
    pub fn deliver_tierup_trap(&mut self, trap: Trap) {
        self.pending_tierup = None;
        self.trap = Some(trap);
    }

    /// Snapshot this vCPU's window (its `[0, prefix_len)` span) after it finishes — the root's image
    /// for capture. (The bytes also live in the shared backing the host handed in, so a wasm host can
    /// read them straight from the `SharedArrayBuffer` instead.)
    pub fn snapshot(&self, prefix_len: u64) -> Vec<u8> {
        self.mem
            .as_ref()
            .map(|m| m.snapshot(prefix_len))
            .unwrap_or_default()
    }
}

/// Durability seam (Slice 1c-6): the bytecode mirror of [`crate::run_capture_reserved_with_host`] —
/// seed the window with `init_mem` (which for a durable run carries the state word + shadow region),
/// run `m`'s transformed entry over a caller-prepared `host` (the powerbox), and snapshot the window
/// (the `SNAP_CAP` span, matching the tree-walker / JIT durable capture). Single-vCPU, single-fiber
/// freeze/thaw is **driven entirely by the transform's emitted IR** — the engine just runs it; this
/// is the entry the freeze/thaw harness (`bytecode_durable.rs`) and the `super::run_with_host_fast`
/// fast path use. `None` if the module is outside the engine's subset.
pub fn compile_and_run_capture_reserved_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> Option<Capture> {
    // Multi-vCPU durability (`thread.*`) is out of scope: a durable thread spawn needs the
    // multi-worker freeze the engine doesn't drive, so always refuse it (the caller falls back to the
    // tree-walker), lest it write a silently-wrong artifact. §14 **nesting** (`Instantiator`
    // cap.calls) is likewise out of scope (DURABILITY.md §4): the tree-walker owns the durable
    // nesting rules — the freezable-module admission check, child durability inheritance, and the
    // fail-closed refusal of a freeze over a live-or-unjoined §14 child; this engine's own
    // instantiate arm has none of them, so driving a durable §14 module here would both skip the
    // admission rule and mint the exact thaw-faulting artifact the tree-walker refuses.
    let outside = m.funcs.iter().flat_map(|f| f.blocks.iter()).any(|b| {
        b.insts.iter().any(|i| {
            matches!(i, Inst::ThreadSpawn { .. } | Inst::ThreadJoin { .. })
                || matches!(i, Inst::CapCall { type_id, .. } if *type_id == super::cap_id::INSTANTIATOR)
        })
    });
    if outside {
        return None;
    }
    // `cont.*` durability is fully supported (DURABILITY.md §12.8): the per-fiber shadow-SP swap keeps
    // the active word on the running context (so a freeze poll spills into the right region), the freeze
    // driver flattens idle parked fibers into their regions, and thaw seeding re-creates them from the
    // artifact residue. So a single-vCPU `cont.*` module is driven here in any window state (NORMAL /
    // UNWINDING freeze / REWINDING thaw); only multi-vCPU `thread.*` (above) still falls back.
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_window(super::SNAP_CAP))
        .unwrap_or_default();
    Some((r, snap))
}

/// An [`ir_trace`] result: the executed instruction-location sequence plus the run's result.
pub type IrTrace = (Vec<super::IrPc>, Result<Vec<Value>, Trap>);

/// A per-step **window-variable** trace ([`ir_window_trace`]): each executed instruction's [`crate::IrPc`]
/// paired with the watched window range's bytes at that point, plus the run result.
pub type WindowTrace = (Vec<(super::IrPc, Vec<u8>)>, Result<Vec<Value>, Trap>);

/// A per-step **SSA-value** trace ([`ir_value_trace`]): each executed instruction's [`crate::IrPc`]
/// paired with the current frame's typed block-local SSA values, plus the run result.
pub type ValueTrace = (Vec<(super::IrPc, Vec<Value>)>, Result<Vec<Value>, Trap>);

/// Debug seam (Slice 1c-3): single-step `m`'s `func(args)` and record the [`crate::IrPc`] of each
/// **instruction** executed (terminators are skipped, matching the tree-walker's `before_op`, which
/// only stops at instructions), returning the location trace plus the result. `None` if the module is
/// outside the engine's subset, or if a step hits a concurrency/coroutine seam (debug is single-vCPU,
/// seam-free — DEBUGGING.md S4). Stepping uses `budget = 1` so each `resume` runs exactly one op.
///
/// The resulting trace is **identical** to driving the tree-walker [`crate::Inspector`] with
/// `seek(0), seek(1), …` — that equality (checked by `bytecode_debug.rs`) is what proves the engine
/// reports tree-walker-identical locations, so breakpoints/stepping at [`crate::IrPc`] granularity
/// land at the same program points on both backends.
pub fn ir_trace(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Option<IrTrace> {
    let c = compile_module_unfused(&m.funcs)?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            trace.push(pc);
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Debug-seam **variable-inspection** support (DEBUGGING.md §1b G2). Like [`ir_trace`], but at each
/// instruction step also snapshots `len` window bytes at `addr` — the value a *window-located* source
/// variable (`VarLoc::Window`) holds at that program point. Register-allocated SSA values have no
/// stable cross-engine storage (the bytecode engine packs them into reused slots), but a window
/// variable lives at a shared address in the same `Mem` both engines drive, so its value *is*
/// comparable per step. Paired with the tree-walker `Inspector` driven by `seek(t)` +
/// `read_var`/`read_window`, this proves the two engines hold the **same variable value at every
/// step** — not merely the same locations (`ir_trace`). `None` on the same out-of-subset / seam
/// conditions as [`ir_trace`]. Test surface; not a production entry point.
pub fn ir_window_trace(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    addr: u64,
    len: usize,
) -> Option<WindowTrace> {
    let c = compile_module_unfused(&m.funcs)?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        // Snapshot the window var *before* running the op — the same point `Inspector::seek(t)` pauses
        // at (paused before the op at clock `t`), so the two byte sequences align step-for-step.
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            let bytes = mem
                .as_ref()
                .and_then(|mm| mm.read_window(addr, len).ok())
                .unwrap_or_default();
            trace.push((pc, bytes));
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue,
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Debug-seam **SSA-value inspection** support (DEBUGGING.md §1b G2). Like [`ir_trace`], but at each
/// instruction step also records the current frame's typed block-local SSA values. `compile_func`
/// assigns a **stable, unique slot per value** (no register reuse / coalescing — "global slot per
/// value"), so an SSA value *is* directly inspectable: `regs[base + i]` typed by `func_value_types`,
/// exactly the storage the tree-walker's `read_ir_value` reads. **Single-block functions only**, where
/// the bytecode slot index equals the tree-walker's block-local value index (both `base`-0); `None`
/// for a multi-block function (per-block slot base differs) or the out-of-subset / seam cases
/// [`ir_trace`] declines. Paired with `Inspector::read_ir_value`/`read_var`, this proves SSA-located
/// variables hold the same value on both engines — the bytecode tier is inspectable, not precluded.
/// Test surface; not a production entry point.
pub fn ir_value_trace(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
) -> Option<ValueTrace> {
    // Single-block scope keeps slot index == tree-walker block-local index (see doc).
    if m.funcs.get(func as usize)?.blocks.len() != 1 {
        return None;
    }
    let types0 =
        svm_verify::func_value_types(&m.funcs[func as usize], &m.funcs, m.memory.is_some())
            .into_iter()
            .next()
            .unwrap_or_default();
    let c = compile_module_unfused(&m.funcs)?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            // The block-0 register window typed per value — the same `(base + i, type)` resolution the
            // tree-walker uses for `read_ir_value`. A not-yet-computed slot reads as its default `Reg`;
            // the caller compares only the defined prefix (where `read_ir_value` returns `Some`).
            let vals: Vec<Value> = types0
                .iter()
                .enumerate()
                .map(|(i, &ty)| vm.regs[vm.base + i].to_value(ty))
                .collect();
            trace.push((pc, vals));
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue,
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Per-module §6 debug metadata a [`FrameReader`] resolves a source variable against: the `-g` info plus
/// the per-`(func, block)` slot base and value types needed to read any live frame's values. Module 0's
/// lives directly on the [`DebugRun`]/[`ScheduledDebugRun`]; a §14 **separate-module** child carries its
/// own here (built from the granted `Module` at spawn, keyed by its pushed source index) so
/// `read_var`/`var_addr`/`value_in_frame` resolve inside the child's body, not just module 0's.
#[derive(Clone)]
struct ModuleDebug {
    /// The child's index in the shared [`ModuleSource`] (`>= 1`) — matched against a frame's module so a
    /// frame outside this child (e.g. an installed §22 unit) isn't misread against these tables.
    module: usize,
    debug: Option<DebugInfo>,
    fn_block_base: Vec<Vec<u32>>,
    fn_block_types: Vec<Vec<Vec<ValType>>>,
}

impl ModuleDebug {
    /// Build the per-`(func, block)` slot base + value types + §6 debug info for module `m`, pushed to the
    /// shared source at index `module`. Mirrors the module-0 computation in [`DebugRun::new_with_host`], so
    /// a separate-module child is inspected exactly as the primary program is.
    fn build(m: &Module, module: usize) -> ModuleDebug {
        let arities: Vec<usize> = m.funcs.iter().map(|g| g.results.len()).collect();
        let mut fn_block_base = Vec::with_capacity(m.funcs.len());
        let mut fn_block_types = Vec::with_capacity(m.funcs.len());
        for g in &m.funcs {
            let mut base = Vec::with_capacity(g.blocks.len());
            let mut n = 0u32;
            for b in &g.blocks {
                base.push(n);
                n += b.params.len() as u32;
                for inst in &b.insts {
                    n += inst.result_count(&arities) as u32;
                }
            }
            fn_block_base.push(base);
            fn_block_types.push(svm_verify::func_value_types(
                g,
                &m.funcs,
                m.memory.is_some(),
            ));
        }
        ModuleDebug {
            module,
            debug: m.debug_info.clone(),
            fn_block_base,
            fn_block_types,
        }
    }
}

/// A single-vCPU time-travel **checkpoint** (DEBUGGING.md W1): the re-executable state of a
/// [`DebugRun`]'s root continuation at logical time [`clock`](DebugRunSnapshot::clock), so a reverse
/// `seek`/`step_back` on the DAP backend can restart a replay here instead of from clock 0 — bounding
/// the replay to the checkpoint stride. The bytecode counterpart of the tree-walker's `SeekCheckpoint`.
/// Opaque to the backend, which only stores it in a ladder and hands it back to [`DebugRun::restore`].
pub struct DebugRunSnapshot {
    clock: u64,
    /// The active root `Vm` (call stack, register windows, cursor) — a plain deep copy.
    active: Vm,
    /// The active continuation id — `ROOT_FIBER` or the handle of the fiber currently running.
    active_id: usize,
    /// Parked resumers on the §12 fiber resume chain: `(fiber id, its `Vm`, resume-result slot)`. Each
    /// `Vm` shares the one window (snapshotted in `mem`), so cloning is a faithful deep copy.
    chain: Vec<(usize, Vm, u32)>,
    /// The §12 fiber registry (handle = index); reconstructed verbatim on restore.
    fibers: Vec<FiberState>,
    /// The non-primary [`ModuleSource`] units (a §14 **separate-module** coroutine's pushed program), so
    /// `restore` re-pushes them and a coroutine frame's `module >= 1` resolves. Empty for a run with only
    /// same-module coroutines. Cheap `Arc` clones — the compiled units are immutable.
    extra_units: Vec<std::sync::Arc<Compiled>>,
    /// The root window's full memory state — committed bytes **and** page-protection map
    /// ([`Mem::layout_snapshot`]), reinstated via [`Mem::restore_layout`] on restore; `None` for a
    /// memoryless run. Capturing the protection map (not just the prefix bytes) is what admits a
    /// **page-mapping** root (`map`/`unmap`/`protect`/grow). Fibers and same-module coroutines share this
    /// one window (coroutines via a `nested_view` over the same backing), so their bytes ride here too.
    mem: Option<super::MemLayout>,
    /// The host's run-mutable replay substate (cap cursor, captured stdout/stderr, clock).
    host: super::HostReplaySubstate,
}

impl DebugRunSnapshot {
    /// The logical time (op clock) this checkpoint was taken at — the ladder key the backend searches.
    pub fn clock(&self) -> u64 {
        self.clock
    }
}

/// Whether a §14 child window (a coroutine or an `instantiate` env) is captured by a checkpoint: its own
/// page map is capturable ([`Mem::layout_snapshot_safe`] — no §13 region aliasing) **and** its extent
/// lies within the parent's snapshotted prefix ([`Mem::nested_within_prefix`], so its bytes ride in the
/// parent reseed rather than a separate copy). A memoryless child is trivially fine. Shared by the
/// single-vCPU and scheduled checkpointable gates, for both coroutines and `instantiate` children.
fn child_checkpointable(child: Option<&Mem>, parent: Option<&Mem>) -> bool {
    child.is_none_or(|m| {
        m.layout_snapshot_safe() && parent.is_some_and(|p| m.nested_within_prefix(p))
    })
}

/// Capture a live `instantiate`-child [`DbgEnv`] into an [`EnvSnapshot`]: its window geometry
/// (`nested_view` base + size) + host replay substate + fuel + its own page-protection map. Its bytes ride
/// in the shared window snapshot (the view shares the root backing region).
fn env_snapshot(e: &DbgEnv, module: usize) -> EnvSnapshot {
    EnvSnapshot {
        win_base: e.mem.as_ref().map_or(0, |m| m.window.base()),
        size_log2: e
            .mem
            .as_ref()
            .map_or(0, |m| m.window.reserved().trailing_zeros() as u8),
        module,
        host: e.host.replay_substate(),
        fuel: e.fuel,
        prot: e.mem.as_ref().map_or_else(Vec::new, |m| m.prot_snapshot()),
    }
}

/// Rebuild a §14 `instantiate`-child [`DbgEnv`] from an [`EnvSnapshot`] on restore — the inverse of
/// [`env_snapshot`]: recreate its `nested_view` over the (reseeded) `shared_mem` window, its attenuated
/// `Instantiator` + `AddressSpace` powerbox over `[0, child_size)` (deterministic in `child_size`), the
/// captured host replay substate, its natural module-0 table, and its fuel quota.
fn rebuild_env(es: &EnvSnapshot, shared_mem: Option<&Mem>, source: &ModuleSource) -> DbgEnv {
    let child_size = 1u64 << es.size_log2;
    let mem = shared_mem.map(|m| m.nested_view(es.win_base, es.size_log2));
    if let Some(m) = &mem {
        m.install_prot(&es.prot); // its `map`/`unmap`/`protect`ed pages (bytes rode in the shared reseed)
    }
    let progs_len = source.get(es.module).map_or(0, |u| u.progs.len());
    let mut host = Host::new();
    host.grant_instantiator(0, child_size);
    host.grant_address_space(0, child_size);
    host.restore_replay_substate(&es.host);
    DbgEnv {
        mem,
        host,
        table: build_table_for(progs_len, 0, es.module as u32),
        fuel: es.fuel,
    }
}

/// A minimal **resumable bytecode debug session** (DEBUGGING.md §1b G3) — the engine-level primitive a
/// DAP-over-bytecode backend would wire into, the first prerequisite for that second backend. Holds the
/// running [`Vm`] across stops: [`DebugRun::run_to`] steps until the current op's [`crate::IrPc`] is a
/// breakpoint (stopping *before* it, like the tree-walker's `seek`/`run_until_stop`) or the run
/// finishes, and is **resumable** — call it again to reach the next hit (a loop-body breakpoint each
/// iteration). [`DebugRun::value`] reads a block-local SSA value at the current stop, typed via
/// `func_value_types` over the stable per-value slots — the bytecode counterpart of
/// `Inspector::read_ir_value`. Scoped to a single function (the value reader resolves slots for the
/// entry function's blocks; a call or concurrency seam ends the run). Test surface; not production.
pub struct DebugRun {
    source: std::sync::Arc<ModuleSource>,
    table: SharedSlots,
    mem: Option<Mem>,
    host: Host,
    /// The reified continuation being debugged: the active `Vm` plus its §12 fiber resume `chain`. A
    /// `cont.resume` switches `vt.active` into a fiber; `suspend` / a fiber return switches back. The
    /// debugger inspects (backtrace / read_var) the **active** continuation.
    vt: VTask,
    /// The session's §12 fiber registry (handle = index). Populated by `cont.new`; rebuilt
    /// deterministically on a reverse `seek` replay. Empty for a fiber-free program.
    fibers: Vec<FiberState>,
    /// Per-**function**, per-block slot base (mirror of `compile_func`'s `base`) — for reading a value
    /// in any live call frame, not just the innermost.
    fn_block_base: Vec<Vec<u32>>,
    /// Per-function, per-block value types (`func_value_types`), for typing a slot's `Reg` to a `Value`.
    fn_block_types: Vec<Vec<Vec<ValType>>>,
    /// The §6 debug info (cloned from the module), for resolving a source variable name to its `VarLoc`
    /// in [`read_var`](DebugRun::read_var). `None` ⇒ the module carried no `-g` section.
    debug: Option<DebugInfo>,
    /// Paused on a reported breakpoint — step past it before the next `run_to` so we make progress.
    at_bp: bool,
    done: Option<Result<Vec<Value>, Trap>>,
    /// Number of ops executed so far — the **logical clock** for reverse debugging (DEBUGGING.md W1).
    /// `seek(t)` reaches a state by replaying a fresh run to this count; `step_back` = `seek(clock-1)`.
    op_clock: u64,
    /// The IR functions (for looking up the op about to execute, to compute its memory access when a
    /// watchpoint is armed). `Arc` so `seek`'s replay-rebuild is cheap.
    funcs: std::sync::Arc<[Func]>,
    /// Armed window watchpoints (DEBUGGING.md W2): `(addr, len, kind)`. Empty in the common case, so
    /// the per-op `access_of` computation is skipped entirely. Ids are owned by the caller (the DAP
    /// backend), which re-applies the set after a `seek` rebuild.
    watchpoints: Vec<(u64, u64, super::WatchKind)>,
    /// Set when the last advance parked at a **blocking-stdin** `read` ([`Outcome::StdinPark`],
    /// INTERACTIVE_EMBEDDING.md W4): the read did not execute and `op_clock` did not advance.
    /// Cleared at each advance entry — the parked read re-executes on resume, so the state
    /// re-derives (re-parks or proceeds) rather than being carried (invariant 7).
    stdin_parked: bool,
    /// The session's optional per-op access sink ([`AccessSinkFn`]) — fired before every module-0
    /// op with the op's [`MemEvent`](super::MemEvent), the run's `op_clock`, and task 0. `None`
    /// (the default) is zero-cost. Not part of snapshots; the DAP backend re-installs it on every
    /// `seek` rebuild (the `watch_specs` pattern) and leaves its rev-trace probes silent.
    access_sink: Option<AccessSinkFn>,
    /// The session's **scheduled debugger writes** ([`ScheduledWrite`], slice 8), sorted by clock,
    /// with the cursor of the next un-applied entry. Empty (the default) is one index compare per
    /// advance; the DAP backend re-installs the list on every rebuild.
    scheduled_writes: Vec<(u64, ScheduledWrite)>,
    write_cursor: usize,
    /// Set when [`run_to`](DebugRun::run_to) stopped *before* an op that hits a watchpoint (the access
    /// hasn't applied yet); taken by the caller to report `StopReason::Watchpoint`.
    last_watch: Option<(u64, bool)>,
}

/// The watched range the op at module-0 `(func, block, inst)` would hit, from the live block-local
/// values — the bytecode counterpart of the tree-walker's `access_of` + `watch_hit`. `None` if the op
/// accesses no watched range (or its address can't be resolved). A free fn (not a method) so it borrows
/// only the pieces `run_to` has already split out of `&mut self`.
#[allow(clippy::too_many_arguments)]
fn watch_hit_before(
    vm: &Vm,
    mem: &Option<Mem>,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    watchpoints: &[(u64, u64, super::WatchKind)],
    func: FuncIdx,
    block: usize,
    inst: usize,
) -> Option<(u64, bool)> {
    let ir_inst = funcs
        .get(func as usize)?
        .blocks
        .get(block)?
        .insts
        .get(inst)?;
    let base_off = *fn_block_base.get(func as usize)?.get(block)? as usize;
    let vals = vm.regs.get(vm.base + base_off..)?;
    // `watch_accesses` (not `access_of`): bulk `mem.copy`/`mem.move`/`mem.fill` and v128 ops
    // check both their spans, so a memcpy over a watched byte stops here like a plain store.
    super::watch_accesses(ir_inst, vals, mem)
        .into_iter()
        .find_map(|acc| {
            let super::MemAccess::Range { base, width, write } = acc else {
                return None;
            };
            let end = base.saturating_add(width as u64);
            watchpoints.iter().find_map(|(addr, len, kind)| {
                let w_end = addr.saturating_add(*len);
                (base < w_end && *addr < end && kind.fires_on(write)).then_some((base, write))
            })
        })
}

/// A debug-session **access sink** (INTERACTIVE_EMBEDDING.md slice 3): observes every module-0
/// memory op the session is about to execute — `(clock-or-turn, task, event)`, with **raw
/// pre-confinement addresses** (the W3 hook-pass vocabulary, [`super::MemEvent`]) — with **no
/// module rewrite**, so the machine view, SSA slots, and the op-clock are identical with a sink
/// installed or absent (invariant 9b: observation never perturbs semantics). Zero cost when
/// absent (callers gate on `Some`). Fed to host-side models (cache/paging/shared-state) by the
/// DAP backend.
pub type AccessSinkFn = Box<dyn FnMut(u64, usize, super::MemEvent) + Send>;

/// The window memory-map introspection tuple — `(page_size, mapped, reserved, explicit-state
/// pages)`, the shape `Mem::map_info` returns (INTERACTIVE_EMBEDDING.md slice 5).
pub type MemMapInfo = (u64, u64, u64, Vec<(u64, u8)>);

/// Build the #750 paged-driver **page-state table** from a window's [`MemMapInfo`]: one byte per
/// page over `[0, coverage)` — `0 = Unmapped`, `1 = Rw`, `2 = Ro` (the emitted check's encoding) —
/// where `coverage` (also returned) is `max(mapped prefix, highest explicit entry end)` in bytes.
///
/// This is THE per-emitted-call driver contract for a page-checked run (refresh from
/// [`Vcpu::mem_map_info`], write the table where emitted code can read it, its base to the
/// `"pagestate"` global, and **`coverage` — not the reserved mask-domain size — to `"mapped"`**):
/// the bound check then traps everything above the table exactly where the interpreter (no
/// entries above) faults, and the page states refine within. Returning the coverage alongside the
/// table is what makes the contract hard to get wrong. A `Backed` (§13 region-aliased) page is
/// marked `Unmapped` — fail-closed (the emitted tier cannot read a region's bytes), and
/// unreachable for a paged module anyway (SharedRegion gates the whole module off the paged tier).
pub fn build_pagestate_table(info: &MemMapInfo) -> (Vec<u8>, u64) {
    let (page, mapped, _reserved, entries) = info;
    let top = entries
        .iter()
        .map(|(off, _)| off / page + 1)
        .max()
        .unwrap_or(0)
        .max(mapped / page);
    let mut t = vec![0u8; top as usize];
    for (i, b) in t.iter_mut().enumerate() {
        if (i as u64) * page < *mapped {
            *b = 1; // Rw default inside the mapped prefix
        }
    }
    for (off, kind) in entries {
        // `map_info` kinds: 0 = Ro, 1 = Rw, 2 = Unmapped, 3 = Backed (§13 alias).
        t[(off / page) as usize] = match kind {
            0 => 2,
            1 => 1,
            _ => 0, // Unmapped, and Backed fail-closed (see above)
        };
    }
    let coverage = t.len() as u64 * page;
    (t, coverage)
}

/// Decode + report the op the active continuation is about to execute to `sink` (module-0 ops
/// only, like the watchpoint scan; coroutine-child ops over their own confined windows are out of
/// scope). The decode is the same live-SSA lookup as [`watch_hit_before`]; the event vocabulary
/// and address semantics are the instrumentation pass's, pinned by the `access_sink_diff`
/// differential.
fn emit_access(
    vm: &Vm,
    source: &ModuleSource,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    clock: u64,
    task: usize,
    sink: &mut AccessSinkFn,
) {
    let Some(pc) = vm.cur_ir_pc(source) else {
        return;
    };
    if pc.module != 0 {
        return;
    }
    let Some(ir_inst) = funcs
        .get(pc.func as usize)
        .and_then(|f| f.blocks.get(pc.block))
        .and_then(|b| b.insts.get(pc.inst))
    else {
        return;
    };
    let Some(base_off) = fn_block_base
        .get(pc.func as usize)
        .and_then(|v| v.get(pc.block))
    else {
        return;
    };
    let Some(vals) = vm.regs.get(vm.base + *base_off as usize..) else {
        return;
    };
    if let Some(ev) = super::mem_event_of(ir_inst, vals) {
        sink(clock, task, ev);
    }
}

/// The outcome of advancing a debug session's active continuation by one op ([`debug_advance_fiber`]).
enum FiberStep {
    /// One op ran (a normal op, or a `cont.*` / fiber-return switch) — the clock ticks, keep going.
    Stepped,
    /// The **root** activation of this continuation returned (`chain` empty) — its result.
    Finished(Vec<Value>),
    /// A trap (including a `FiberFault`).
    Trapped(Trap),
    /// A non-fiber seam the caller must apply: `thread.spawn`/`join`, `memory.wait`/`notify`,
    /// `instantiate`, coroutine, tier-up. The single-vCPU [`DebugRun`] treats these as `Malformed`; the
    /// multi-vCPU [`ScheduledDebugRun`] dispatches the ones it schedules (spawn/join/wait/notify).
    Other(Outcome),
}

/// Run **one op** of a debug session's active continuation (`vt.active`), applying any §12 fiber switch
/// (`cont.new` registers a fiber in the run-shared `fibers`, `cont.resume` switches into one, `suspend`
/// / a fiber's return switches back). Non-fiber seams are handed back as [`FiberStep::Other`]. The debug
/// counterpart of [`step_vcpu`]'s fiber handling, minus durability (debug runs are non-durable, so no
/// `shadow_switch` / `fiber_sp`). `fibers` is run-shared (a fiber created on one vCPU can be resumed on
/// another — D57 migration) and rebuilt deterministically on a reverse `seek` replay.
fn debug_advance_fiber(
    vt: &mut VTask,
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    // Step-into a §14 coroutine body (single-vCPU `DebugRun` only): while a coroutine child is the
    // its **own** confined `mem`/`host`/`table`, the op-by-op counterpart of `resume_coro`. Surfacing
    // each child op is what makes breakpoints fire inside the body and the child frame inspectable.
    // Stepping *inside* a §22 `Jit.invoke`d unit (single-vCPU step-into): drive it one op over the
    // caller's shared window/table (the §22 counterpart of `active_coro`), so a breakpoint fires inside.
    if vt.active_invoke.is_some() {
        return step_active_invoke(vt, source, table, fuel, mem, host);
    }
    match vt
        .active
        .resume(source, table, fuel, mem, &mut HostCell::Excl(host), 1)
    {
        Ok(Outcome::Suspended) => FiberStep::Stepped,
        Ok(Outcome::Done(vals)) => match vt.chain.pop() {
            // The root activation finished — the run's result.
            None => FiberStep::Finished(vals),
            // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` to its resumer.
            Some((rid, resumer, rdst)) => {
                fibers[vt.active_id] = FiberState::Done;
                let retval = vals.first().copied().unwrap_or(Value::I64(0));
                vt.active = resumer;
                vt.active_id = rid;
                vt.active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                vt.active.set(rdst + 1, Reg::from_value(retval));
                FiberStep::Stepped
            }
        },
        Ok(Outcome::ContNew { funcref, sp, dst }) => {
            if fibers.len() + 1 >= super::MAX_FIBERS {
                return FiberStep::Trapped(Trap::FiberFault);
            }
            let h = fibers.len() as i32;
            fibers.push(FiberState::Pending { funcref, sp });
            vt.active.set(dst, Reg::from_i32(h));
            FiberStep::Stepped
        }
        // This stepper runs where fibers never event-park (it traps on `WaitParked`/`CapParked`
        // below), so the I48 `blocking` flag is a no-op here — a blocking resume of a fiber that
        // only ever suspends/returns behaves exactly like `cont.resume`.
        Ok(Outcome::ContResume {
            kh,
            arg,
            dst,
            blocking: _,
            resume_ip: _,
        }) => {
            let k = kh as usize;
            let target = match fibers.get_mut(k) {
                Some(slot @ FiberState::Pending { .. }) => {
                    let (funcref, sp) =
                        match std::mem::replace(slot, FiberState::Running { blocking_ip: None }) {
                            FiberState::Pending { funcref, sp } => (funcref, sp),
                            _ => unreachable!(),
                        };
                    let m0 = source.primary();
                    let f = (funcref as u32 as usize) & m0.table_mask;
                    let ok = m0
                        .sigs
                        .get(f)
                        .is_some_and(|(p, r)| p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS);
                    if !ok {
                        return FiberStep::Trapped(Trap::FiberFault);
                    }
                    match Vm::new(&m0, f, &[Value::I64(sp), Value::I64(arg)]) {
                        Ok(v) => v,
                        Err(t) => return FiberStep::Trapped(t),
                    }
                }
                Some(slot @ FiberState::Parked { .. }) => {
                    match std::mem::replace(slot, FiberState::Running { blocking_ip: None }) {
                        FiberState::Parked {
                            mut vm,
                            suspend_dst,
                        } => {
                            vm.set(suspend_dst, Reg::from_i64(arg));
                            vm
                        }
                        _ => unreachable!(),
                    }
                }
                _ => return FiberStep::Trapped(Trap::FiberFault), // forged / Running / Done
            };
            let resumer = std::mem::replace(&mut vt.active, target);
            vt.chain.push((vt.active_id, resumer, dst));
            vt.active_id = k;
            FiberStep::Stepped
        }
        Ok(Outcome::FiberSuspend { value, dst }) => {
            // Pop the resumer to switch back to; an empty chain means the root tried to `suspend`.
            let Some((rid, resumer, rdst)) = vt.chain.pop() else {
                return FiberStep::Trapped(Trap::FiberFault);
            };
            let suspended = std::mem::replace(&mut vt.active, resumer);
            fibers[vt.active_id] = FiberState::Parked {
                vm: suspended,
                suspend_dst: dst,
            };
            vt.active_id = rid;
            vt.active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
            vt.active.set(rdst + 1, Reg::from_i64(value));
            FiberStep::Stepped
        }
        // §22 guest-JIT install / uninstall / invoke: self-contained host-side ops (they mutate only
        // `vt.active` + the shared dispatch table, spawning no scheduler task), so — like the coroutine
        // arms above — they are serviced **inline** here, which is why BOTH the single-vCPU `DebugRun`
        // and the `ScheduledDebugRun` reach them (the `FiberStep::Other` decline sites never see a Jit
        // outcome). `invoke` runs the unit to completion as a seam-free leaf (stepping *over* it, not
        // into it — matching production `run_invoke`). A forged handle / out-of-coverage unit traps the
        // vCPU (`CapFault`/`Malformed`), exactly as the production `drive`. (DESIGN.md §22 debug tier.)
        Ok(Outcome::JitInstall { h, code, dst }) => {
            match dbg_jit_install(vt, host, source, table, h, code, dst) {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        Ok(Outcome::JitUninstall { h, slot, dst }) => {
            match dbg_jit_uninstall(vt, host, source, table, h, slot, dst) {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        Ok(Outcome::JitInvoke {
            h,
            code,
            argv,
            dst,
            params,
            results,
        }) => {
            // Single-vCPU `DebugRun` (`invoke_step_into`) steps *into* the invoked unit; the scheduled
            // engine keeps it an opaque leaf. Either way it starts here — the step-into arm arms
            // `active_invoke` and the next advance steps the unit's first op.
            let r = if vt.invoke_step_into {
                dbg_jit_invoke_step_into(vt, host, source, h, code, &argv, dst, &params, &results)
            } else {
                dbg_jit_invoke_leaf(
                    vt, host, source, table, fuel, mem, h, code, &argv, dst, &params, &results,
                )
            };
            match r {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        // F2 — a punted host call in a debug advance keeps the pre-F2 inline wait (the debug
        // drivers' whole-vCPU-park shape is sanctioned tiering, invariant 9 observability
        // corollary; checkpointing across one is already excluded by `checkpoint_safe`'s
        // replay-substate rules — the wait happens inside the advance, leaving no parked state).
        Ok(Outcome::CapPending { id, dst }) => {
            let r = host.completions().wait(id);
            vt.active.set(dst, Reg::from_i64(r));
            FiberStep::Stepped
        }
        // Threads / wait / notify / instantiate / (scheduled-engine) separate-module coroutine / tier-up
        // — a scheduler seam the caller applies (single-vCPU `DebugRun` rejects them; the scheduled engine
        // dispatches its subset).
        Ok(other) => FiberStep::Other(other),
        Err(t) => FiberStep::Trapped(t),
    }
}

/// A read-only inspection view over **one vCPU's** reified state (`vm`) plus the module's §6 debug
/// metadata. This is the shared frame-reading engine behind both the single-vCPU [`DebugRun`] and the
/// multi-vCPU [`ScheduledDebugRun`]: given any task's `Vm`, it resolves backtrace frames, block-local
/// SSA values, and named source variables identically — so a thread selected mid-stop (`select_task`)
/// reads its own stack through the exact same code the single-vCPU path uses.
struct FrameReader<'a> {
    vm: &'a Vm,
    source: &'a ModuleSource,
    mem: &'a Option<Mem>,
    debug: Option<&'a DebugInfo>,
    fn_block_base: &'a [Vec<u32>],
    fn_block_types: &'a [Vec<Vec<ValType>>],
    /// The active §14 **separate-module** coroutine's own §6 metadata (module index `>= 1`), set while
    /// stepping inside its body — so a frame in the child resolves against the child's funcs. `None` when
    /// the active continuation is module 0 (the parent or a same-module coroutine), where the module-0
    /// fields above apply. See [`FrameReader::md_for`].
    coro_debug: Option<&'a ModuleDebug>,
}

/// A resolved write destination (slice 8): a typed absolute regs slot (a promoted SSA scalar) or
/// a confined window address (a memory-located variable).
enum WriteTarget {
    Ssa { reg: usize, ty: ValType },
    Win { addr: u64 },
}

/// A **debugger write scheduled at a clock/turn** (INTERACTIVE_EMBEDDING.md slice 8): re-applied
/// whenever execution passes that clock on **any** path — a live resume and a seek replay reach
/// identical states, which is what keeps the history slider truthful after an edit. `task` names
/// the focused vCPU a `Var` write resolves in on the scheduled engine (ignored single-vCPU).
#[derive(Clone, Debug)]
pub enum ScheduledWrite {
    Window {
        addr: u64,
        bytes: Vec<u8>,
    },
    Var {
        task: usize,
        frame: usize,
        name: String,
        value: i64,
        width: usize,
    },
}

/// Coerce + store `value` into the typed regs slot / window target. Best-effort like the live
/// write: an unresolvable or float target is skipped.
fn apply_target(
    target: Option<WriteTarget>,
    value: i64,
    width: usize,
    vm_regs: &mut [Reg],
    mem: &mut Option<Mem>,
) {
    match target {
        Some(WriteTarget::Ssa { reg, ty }) => {
            let v = match ty {
                ValType::I32 => Value::I32(value as i32),
                ValType::I64 => Value::I64(value),
                _ => return,
            };
            if let Some(r) = vm_regs.get_mut(reg) {
                *r = Reg::from_value(v);
            }
        }
        Some(WriteTarget::Win { addr }) => {
            let w = width.clamp(1, 8);
            if let Some(m) = mem.as_mut() {
                let _ = m.write_bytes(addr, &value.to_le_bytes()[..w]);
            }
        }
        None => {}
    }
}

/// Apply every scheduled write due at `clock` to a **single-vCPU** run's pieces; `cursor` advances
/// past applied and stale entries (entries below `clock` are inside a restored checkpoint already).
#[allow(clippy::too_many_arguments)]
fn apply_due_writes(
    writes: &[(u64, ScheduledWrite)],
    cursor: &mut usize,
    clock: u64,
    vt: &mut VTask,
    source: &ModuleSource,
    mem: &mut Option<Mem>,
    debug: Option<&DebugInfo>,
    fn_block_base: &[Vec<u32>],
    fn_block_types: &[Vec<Vec<ValType>>],
) {
    while *cursor < writes.len() && writes[*cursor].0 < clock {
        *cursor += 1;
    }
    while *cursor < writes.len() && writes[*cursor].0 == clock {
        match &writes[*cursor].1 {
            ScheduledWrite::Window { addr, bytes } => {
                if let Some(m) = mem.as_mut() {
                    let _ = m.write_bytes(*addr, bytes);
                }
            }
            ScheduledWrite::Var {
                frame,
                name,
                value,
                width,
                ..
            } => {
                if vt.active_invoke.is_none() {
                    let target = FrameReader {
                        vm: &vt.active,
                        source,
                        mem: &*mem,
                        debug,
                        fn_block_base,
                        fn_block_types,
                        coro_debug: None,
                    }
                    .write_target(*frame, name);
                    apply_target(target, *value, *width, &mut vt.active.regs, mem);
                }
            }
        }
        *cursor += 1;
    }
}

/// The scheduled-engine twin of [`apply_due_writes`]: a `Var` write resolves in its recorded
/// `task`'s frame.
#[allow(clippy::too_many_arguments)]
fn apply_due_writes_sched(
    writes: &[(u64, ScheduledWrite)],
    cursor: &mut usize,
    turn: u64,
    tasks: &mut [DbgTask],
    source: &ModuleSource,
    mem: &mut Option<Mem>,
    debug: Option<&DebugInfo>,
    fn_block_base: &[Vec<u32>],
    fn_block_types: &[Vec<Vec<ValType>>],
) {
    while *cursor < writes.len() && writes[*cursor].0 < turn {
        *cursor += 1;
    }
    while *cursor < writes.len() && writes[*cursor].0 == turn {
        match &writes[*cursor].1 {
            ScheduledWrite::Window { addr, bytes } => {
                if let Some(m) = mem.as_mut() {
                    let _ = m.write_bytes(*addr, bytes);
                }
            }
            ScheduledWrite::Var {
                task,
                frame,
                name,
                value,
                width,
            } => {
                if let Some(t) = tasks.get_mut(*task) {
                    {
                        let target = FrameReader {
                            vm: &t.vt.active,
                            source,
                            mem: &*mem,
                            debug,
                            fn_block_base,
                            fn_block_types,
                            coro_debug: None,
                        }
                        .write_target(*frame, name);
                        apply_target(target, *value, *width, &mut t.vt.active.regs, mem);
                    }
                }
            }
        }
        *cursor += 1;
    }
}

impl<'a> FrameReader<'a> {
    /// Call-stack depth (running activation + suspended callers).
    fn depth(&self) -> usize {
        self.vm.stack.len() + 1
    }

    /// The `(§6 debug, per-block slot base, per-block value types)` to read a frame in `module` against:
    /// module 0's from the session fields, or an active separate-module coroutine's own. `None` for any
    /// other module (e.g. an installed §22 unit whose frames carry no debug tables here) — such a frame is
    /// not source-inspectable, matching the pre-slice module-0 gate.
    #[allow(clippy::type_complexity)]
    fn md_for(
        &self,
        module: usize,
    ) -> Option<(
        Option<&'a DebugInfo>,
        &'a [Vec<u32>],
        &'a [Vec<Vec<ValType>>],
    )> {
        if module == 0 {
            return Some((self.debug, self.fn_block_base, self.fn_block_types));
        }
        let md = self.coro_debug?;
        (md.module == module).then_some((md.debug.as_ref(), &md.fn_block_base, &md.fn_block_types))
    }

    /// The `(module, func, block, inst, window base)` of the frame `depth` levels from the top (0 =
    /// running activation; each caller resolved at its call site, `resume_pc - 1`). `None` past the
    /// stack or when the top is paused on a non-instruction.
    fn frame_at(&self, depth: usize) -> Option<(usize, usize, usize, usize, usize)> {
        if depth == 0 {
            let pc = self.vm.cur_ir_pc(self.source)?;
            return Some((self.vm.module, self.vm.cur, pc.block, pc.inst, self.vm.base));
        }
        let n = self.vm.stack.len();
        let &(module, f, base, resume_pc, _) = self.vm.stack.get(n.checked_sub(depth)?)?;
        let cm = self.source.get(module)?;
        let (block, inst) = cm
            .progs
            .get(f)?
            .src
            .get(resume_pc.checked_sub(1)?)
            .copied()
            .flatten()?;
        Some((module, f, block as usize, inst as usize, base))
    }

    /// The `IrPc` of the frame `depth` levels from the top.
    fn frame_pc(&self, depth: usize) -> Option<super::IrPc> {
        let (module, func, block, inst, _) = self.frame_at(depth)?;
        Some(super::IrPc {
            module: module as u32,
            func: func as FuncIdx,
            block,
            inst,
        })
    }

    /// Block-local SSA value `idx` in the frame `depth` levels from the top, typed against that frame's
    /// module (module 0's, or an active separate-module coroutine's own).
    fn value_in_frame(&self, depth: usize, idx: usize) -> Option<Value> {
        let (module, func, block, _inst, base) = self.frame_at(depth)?;
        let (_, fn_block_base, fn_block_types) = self.md_for(module)?;
        let off = *fn_block_base.get(func)?.get(block)? as usize;
        let ty = *fn_block_types.get(func)?.get(block)?.get(idx)?;
        Some(self.vm.regs[base + off + idx].to_value(ty))
    }

    /// Where a **write** to source variable `name` in frame `depth` lands (slice 8): the absolute
    /// regs slot + type for a promoted SSA scalar, or the confined window address for a
    /// memory-located var — the write-side mirror of [`FrameReader::read_var`]'s resolution.
    fn write_target(&self, depth: usize, name: &str) -> Option<WriteTarget> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let (di, fn_block_base, fn_block_types) = self.md_for(module)?;
        let var = super::pick_var(di?, func as FuncIdx, name, block, inst)?;
        let off = *fn_block_base.get(func)?.get(block)? as usize;
        let slot = |idx: usize| -> Option<WriteTarget> {
            let ty = *fn_block_types.get(func)?.get(block)?.get(idx)?;
            Some(WriteTarget::Ssa {
                reg: base + off + idx,
                ty,
            })
        };
        match &var.loc {
            VarLoc::Ssa { value } => slot(*value as usize),
            VarLoc::SsaList(locs) => slot(super::loclist_value(locs, block, inst)? as usize),
            VarLoc::Window { off: o } => Some(WriteTarget::Win {
                addr: (self.vm.regs[base].i64() as u64).wrapping_add(*o as u64),
            }),
            VarLoc::WindowVia { base: locs, off: o } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                Some(WriteTarget::Win {
                    addr: addr.wrapping_add(*o as u64),
                })
            }
            VarLoc::Fixed { addr } => Some(WriteTarget::Win { addr: *addr }),
        }
    }

    /// Read a source variable by name in the frame `depth` levels from the top, resolving its `VarLoc`
    /// over that frame's module's §6 debug info (SSA slot / window / fixed) — module 0's, or an active
    /// separate-module coroutine's own. `None` if unresolvable here.
    fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let di = self.md_for(module)?.0?;
        let var = super::pick_var(di, func as FuncIdx, name, block, inst)?;
        let window_read = |addr: u64| -> Option<VarValue> {
            Some(VarValue::Bytes(
                self.mem.as_ref()?.read_window(addr, width).ok()?,
            ))
        };
        match &var.loc {
            VarLoc::Ssa { value } => self
                .value_in_frame(depth, *value as usize)
                .map(VarValue::Value),
            VarLoc::SsaList(locs) => {
                let v = super::loclist_value(locs, block, inst)?;
                self.value_in_frame(depth, v as usize).map(VarValue::Value)
            }
            // Address = data-SP (the frame's first value, v0) + off.
            VarLoc::Window { off } => {
                window_read((self.vm.regs[base].i64() as u64).wrapping_add(*off as u64))
            }
            VarLoc::WindowVia { base: locs, off } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                window_read(addr.wrapping_add(*off as u64))
            }
            VarLoc::Fixed { addr } => window_read(*addr),
        }
    }

    /// The window address of a memory-located source variable by name in the frame `depth` from the
    /// top; `None` for a promoted SSA scalar (no address) or an unresolvable name. Resolves against that
    /// frame's module's §6 info (module 0's, or an active separate-module coroutine's own).
    fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let di = self.md_for(module)?.0?;
        let var = super::pick_var(di, func as FuncIdx, name, block, inst)?;
        match &var.loc {
            VarLoc::Ssa { .. } | VarLoc::SsaList(_) => None,
            VarLoc::Window { off } => {
                Some((self.vm.regs[base].i64() as u64).wrapping_add(*off as u64))
            }
            VarLoc::WindowVia { base: locs, off } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                Some(addr.wrapping_add(*off as u64))
            }
            VarLoc::Fixed { addr } => Some(*addr),
        }
    }
}

impl DebugRun {
    /// A [`FrameReader`] over this single-vCPU session's currently-stepping `Vm` + debug metadata —
    /// the active §14 coroutine child (over its own confined `mem`) during step-into, else the parent.
    fn reader(&self) -> FrameReader<'_> {
        let vm = self.vt.debug_active();
        FrameReader {
            vm,
            source: &self.source,
            mem: &self.mem,
            debug: self.debug.as_ref(),
            fn_block_base: &self.fn_block_base,
            fn_block_types: &self.fn_block_types,
            // A separate-module coroutine carries its own §6 metadata; a same-module one leaves it `None`
            // (its frames are module 0, read against the session fields above).
            coro_debug: None,
        }
    }

    /// Open a debug session on `m`'s `func(args)`. `None` if the module is outside the engine's subset.
    /// The powerbox is empty (`Host::new()`); use [`DebugRun::new_with_host`] to debug a guest that
    /// needs a granted capability (e.g. a §14 `Instantiator` for coroutines).
    pub fn new(m: &Module, func: FuncIdx, args: &[Value]) -> Option<DebugRun> {
        DebugRun::new_with_host(m, func, args, Host::new())
    }

    /// [`DebugRun::new`] carrying a live powerbox `host`, so synchronous `cap.call`s execute against it
    /// — e.g. a §14 `Instantiator` grant (`host.grant_instantiator(..)`) reaching the guest as an
    /// argument, which makes `spawn_coroutine`/`resume`/`yield` debuggable. `None` if the module is
    /// outside the engine's subset.
    pub fn new_with_host(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        host: Host,
    ) -> Option<DebugRun> {
        m.funcs.get(func as usize)?;
        // Slot base + value types per (function, block), so any frame on the call stack is readable — the
        // module-0 counterpart of a §14 separate-module child's own [`ModuleDebug`].
        let ModuleDebug {
            fn_block_base,
            fn_block_types,
            ..
        } = ModuleDebug::build(m, 0);
        let c = compile_module_unfused(&m.funcs)?; // unfused: debug stepping (Slice 5a)
        let dom = Domain::new(c, host.jit_table_log2());
        let mem = build_mem(m);
        let mut vt = VTask::new(&dom.source.primary(), func as usize, args).ok()?;
        vt.invoke_step_into = true; // single-vCPU engine steps *into* §22 Jit.invoke (scheduled = leaf)
        let Domain { source, table } = dom;
        Some(DebugRun {
            source,
            table,
            mem,
            host,
            vt,
            fibers: Vec::new(),
            fn_block_base,
            fn_block_types,
            debug: m.debug_info.clone(),
            at_bp: false,
            done: None,
            op_clock: 0,
            funcs: std::sync::Arc::from(m.funcs.clone()),
            watchpoints: Vec::new(),
            stdin_parked: false,
            access_sink: None,
            scheduled_writes: Vec::new(),
            write_cursor: 0,
            last_watch: None,
        })
    }

    /// Replace the armed **window watchpoints** (DEBUGGING.md W2) — each `(addr, len, kind)` makes
    /// `run_to` stop *before* any op that accesses `[addr, addr+len)` with a matching read/write kind.
    /// Caller-owned ids; re-applied by the DAP backend after a `seek` rebuild.
    pub fn set_watchpoints(&mut self, ranges: Vec<(u64, u64, super::WatchKind)>) {
        self.watchpoints = ranges;
    }

    /// Take the `(addr, write)` of the watchpoint the last `run_to` stopped before (cleared by the
    /// read), so the caller can report `StopReason::Watchpoint`. `None` if the last stop was a plain
    /// breakpoint / step.
    pub fn take_watch_hit(&mut self) -> Option<(u64, bool)> {
        self.last_watch.take()
    }

    /// Ops executed so far — the reverse-debugging clock ([`DebugRun::op_clock`]).
    pub fn op_clock(&self) -> u64 {
        self.op_clock
    }

    /// Whether the last advance parked at a **blocking-stdin** `read` (W4): the run is live and
    /// resumable, paused at the read, and the clock did not advance. Arm the mode via
    /// [`Host::set_stdin_blocking`](super::Host::set_stdin_blocking) on the run's host.
    pub fn stdin_parked(&self) -> bool {
        self.stdin_parked
    }

    /// Append stdin bytes for a parked blocking `read` ([`Host::push_stdin`](super::Host::push_stdin))
    /// — the next advance re-issues the read against them, and the completed read joins the recorded
    /// cap tape so a later `seek` replays it faithfully.
    pub fn provide_stdin(&mut self, bytes: &[u8]) {
        self.host.push_stdin(bytes);
    }

    /// Install the session's per-op **access sink** ([`AccessSinkFn`]) — observation only, zero
    /// cost when never installed. Replaces any prior sink.
    pub fn set_access_sink(&mut self, sink: AccessSinkFn) {
        self.access_sink = Some(sink);
    }

    /// Install the session's **scheduled debugger writes** (slice 8): sorted by clock; entries at
    /// clocks already passed are skipped (a rebuilt run applies them during its replay instead).
    pub fn set_scheduled_writes(&mut self, mut writes: Vec<(u64, ScheduledWrite)>) {
        writes.sort_by_key(|(c, _)| *c);
        self.write_cursor = writes.partition_point(|(c, _)| *c < self.op_clock);
        self.scheduled_writes = writes;
    }

    /// The run's window memory-map introspection ([`MemMapInfo`]). `None` for a memory-less
    /// module.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// Arm the "paused on a breakpoint" state so the next [`run_to`](DebugRun::run_to) steps past the
    /// current op before scanning — used after a `seek`/replay lands exactly on a breakpoint, so a
    /// forward resume makes progress instead of re-reporting the same stop.
    pub fn arm_breakpoint_skip(&mut self) {
        self.at_bp = true;
    }

    /// Whether this run's state is fully captured by its `VTask` continuation + the window bytes + the
    /// host's replay substate — the subset a single-vCPU time-travel **checkpoint** (W1) snapshots. The
    /// bytecode counterpart of [`VCpu::checkpointable`](super::Inspector). The host has grown no state a
    /// checkpoint can't restore (`checkpoint_safe`) and the shared window has a pristine layout
    /// (`layout_snapshot_safe`). **§12 fibers** are admitted (their `Vm`s share the one window), except an
    /// event-parked (`memory.wait`) fiber whose wall-clock deadline is non-deterministic. **§14
    /// coroutines** are admitted (same-module *or* separate-module, whose pushed unit rides in
    /// `extra_units`), including **demand** (`fault_yields`) and self-page-mapping ones — their own page
    /// map is captured (`layout_snapshot_safe`, no §13 regions) and their bytes ride in the parent
    /// snapshot, provided the child window lies within the parent's captured prefix
    /// (`nested_within_prefix`). A region-aliased child stays outside. Outside this subset the DAP backend
    /// falls back to replay-from-clock-0.
    fn checkpointable(&self) -> bool {
        self.done.is_none()
            && self.host.checkpoint_safe()
            && self.mem.as_ref().is_none_or(|m| m.layout_snapshot_safe())
            // Reverse-replay *across* a §22 `Jit.invoke` step-into is out-of-subset (CONSOLIDATION.md
            // §11 debug boundary): the invoked unit's transient `Vm` + its `source.push`ed module aren't
            // captured here, so a checkpoint mid-invoke can't be restored. A reverse-seek near an invoke
            // replays from an earlier checkpoint, which re-enters the invoke deterministically.
            && self.vt.active_invoke.is_none()
            && !self.fibers.iter().any(|f| {
                matches!(
                    f,
                    FiberState::WaitParked { .. } | FiberState::CapParked { .. }
                )
            })
    }

    /// Snapshot this run's continuation at its current [`op_clock`](DebugRun::op_clock) for the `seek`
    /// checkpoint ladder — `None` if the run is outside the [`checkpointable`](DebugRun::checkpointable)
    /// subset. Deep-copies the full `VTask` (active `Vm`, fiber resume chain, fiber registry, coroutine
    /// children) + the shared window bytes + the host's replay substate + any separate-module coroutine's
    /// pushed source units.
    pub fn snapshot(&self) -> Option<DebugRunSnapshot> {
        if !self.checkpointable() {
            return None;
        }
        Some(DebugRunSnapshot {
            clock: self.op_clock,
            active: self.vt.active.clone(),
            active_id: self.vt.active_id,
            chain: self.vt.chain.clone(),
            fibers: self.fibers.clone(),
            extra_units: self.source.extra_units(),
            mem: self.mem.as_ref().map(|m| m.layout_snapshot()),
            host: self.host.replay_substate(),
        })
    }

    /// Restore a [`snapshot`](DebugRun::snapshot) into this **freshly built** run (its powerbox already
    /// re-created + tape re-armed by the backend), so a subsequent replay resumes exactly at the
    /// snapshot's logical time rather than clock 0. Rebuilds the whole `VTask` (active `Vm`, resume
    /// chain, fiber registry, and each same-module coroutine — its `nested_view` recreated over the
    /// reseeded parent window, its Yielder-only host rebuilt), reseeds the window bytes, restores the
    /// host replay substate, and sets the clock. A separate-module coroutine's pushed source units are
    /// re-pushed first (so its `module` index resolves); `table`/`funcs` for module 0 already match.
    pub fn restore(&mut self, snap: &DebugRunSnapshot) {
        self.vt.active = snap.active.clone();
        self.vt.active_id = snap.active_id;
        self.vt.chain = snap.chain.clone();
        self.fibers = snap.fibers.clone();
        // Re-push any separate-module coroutine's units before rebuilding coroutines (their `module`
        // indices resolve against the source).
        self.source.reset_extra(&snap.extra_units);
        if let (Some(m), Some(layout)) = (self.mem.as_mut(), snap.mem.as_ref()) {
            m.restore_layout(layout);
        }
        self.host.restore_replay_substate(&snap.host);
        self.op_clock = snap.clock;
        self.done = None;
        self.at_bp = false;
        self.stdin_parked = false; // a restored run is not parked; a re-executed read re-parks
    }

    /// Execute **exactly one op** (advancing the clock), for replay-based `seek`. Returns `false` once
    /// the run has finished (its result is then available via [`result`](DebugRun::result)). Unlike the
    /// stepping verbs it does not skip unmapped ops or honor breakpoints — it is the raw time quantum.
    pub fn tick(&mut self, fuel: &mut u64) -> bool {
        if self.done.is_some() {
            return false;
        }
        self.at_bp = false;
        self.stdin_parked = false;
        let Self {
            source,
            table,
            mem,
            host,
            vt,
            fibers,
            done,
            op_clock,
            stdin_parked,
            funcs,
            fn_block_base,
            fn_block_types,
            debug,
            access_sink,
            scheduled_writes,
            write_cursor,
            ..
        } = self;
        apply_due_writes(
            scheduled_writes,
            write_cursor,
            *op_clock,
            vt,
            source,
            mem,
            debug.as_ref(),
            fn_block_base,
            fn_block_types,
        );
        if let Some(sink) = access_sink.as_mut() {
            let cur_vm = vt.debug_active();
            emit_access(cur_vm, source, funcs, fn_block_base, *op_clock, 0, sink);
        }
        match debug_advance_fiber(vt, fibers, source, table, fuel, mem, host) {
            FiberStep::Stepped => {
                *op_clock += 1;
                true
            }
            FiberStep::Finished(vals) => {
                *op_clock += 1;
                *done = Some(Ok(vals));
                false
            }
            FiberStep::Trapped(t) => {
                *done = Some(Err(t));
                false
            }
            // Parked at a blocking-stdin read (W4): the read did not run and the clock holds —
            // the run stays live; the driver pushes bytes and re-ticks to re-issue it.
            FiberStep::Other(Outcome::StdinPark) => {
                *stdin_parked = true;
                false
            }
            // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
            FiberStep::Other(_) => {
                *done = Some(Err(Trap::Malformed));
                false
            }
        }
    }

    /// Run until the current op's `IrPc` is in `bps` (stopping *before* it) or the run finishes; returns
    /// the stop pc, or `None` at completion / a seam. Resumable — a re-entry steps past the last hit.
    pub fn run_to(&mut self, bps: &[super::IrPc], fuel: &mut u64) -> Option<super::IrPc> {
        if self.done.is_some() {
            return None;
        }
        self.stdin_parked = false;
        let Self {
            source,
            table,
            mem,
            host,
            vt,
            fibers,
            at_bp,
            done,
            op_clock,
            fn_block_base,
            fn_block_types,
            debug,
            funcs,
            watchpoints,
            stdin_parked,
            access_sink,
            scheduled_writes,
            write_cursor,
            last_watch,
            ..
        } = self;
        // Step past the breakpoint we last reported, so a re-entry makes progress (loop bodies).
        if *at_bp {
            *at_bp = false;
            apply_due_writes(
                scheduled_writes,
                write_cursor,
                *op_clock,
                vt,
                source,
                mem,
                debug.as_ref(),
                fn_block_base,
                fn_block_types,
            );
            if let Some(sink) = access_sink.as_mut() {
                let cur_vm = vt.debug_active();
                emit_access(cur_vm, source, funcs, fn_block_base, *op_clock, 0, sink);
            }
            match debug_advance_fiber(vt, fibers, source, table, fuel, mem, host) {
                FiberStep::Stepped => *op_clock += 1,
                FiberStep::Finished(vals) => {
                    *op_clock += 1;
                    *done = Some(Ok(vals));
                    return None;
                }
                FiberStep::Trapped(t) => {
                    *done = Some(Err(t));
                    return None;
                }
                // Blocking-stdin park (W4): live and resumable, no clock advance (see `tick`).
                FiberStep::Other(Outcome::StdinPark) => {
                    *stdin_parked = true;
                    return None;
                }
                // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
                FiberStep::Other(_) => {
                    *done = Some(Err(Trap::Malformed));
                    return None;
                }
            }
        }
        loop {
            // Scan the currently-stepping continuation — the active §14 coroutine child (over its own
            // confined window) during step-into, else the parent. A same-module child is module 0, so
            // its ops share the parent's pc space; a breakpoint on the child's function fires here.
            let hit = {
                let cur_vm = vt.debug_active();
                let cur_mem = &*mem;
                match cur_vm.cur_ir_pc(source) {
                    Some(pc) if bps.contains(&pc) => Some((pc, None)),
                    Some(pc) if !watchpoints.is_empty() && pc.module == 0 => watch_hit_before(
                        cur_vm,
                        cur_mem,
                        funcs,
                        fn_block_base,
                        watchpoints,
                        pc.func,
                        pc.block,
                        pc.inst,
                    )
                    .map(|w| (pc, Some(w))),
                    _ => None,
                }
            };
            if let Some((pc, watch)) = hit {
                // A watchpoint stops *before* the access applies (step once to observe the new bytes).
                if let Some(w) = watch {
                    *last_watch = Some(w);
                }
                *at_bp = true;
                return Some(pc);
            }
            apply_due_writes(
                scheduled_writes,
                write_cursor,
                *op_clock,
                vt,
                source,
                mem,
                debug.as_ref(),
                fn_block_base,
                fn_block_types,
            );
            if let Some(sink) = access_sink.as_mut() {
                let cur_vm = vt.debug_active();
                emit_access(cur_vm, source, funcs, fn_block_base, *op_clock, 0, sink);
            }
            match debug_advance_fiber(vt, fibers, source, table, fuel, mem, host) {
                FiberStep::Stepped => {
                    *op_clock += 1;
                    continue;
                }
                FiberStep::Finished(vals) => {
                    *op_clock += 1;
                    *done = Some(Ok(vals));
                    return None;
                }
                FiberStep::Trapped(t) => {
                    *done = Some(Err(t));
                    return None;
                }
                // Blocking-stdin park (W4): live and resumable, no clock advance (see `tick`).
                FiberStep::Other(Outcome::StdinPark) => {
                    *stdin_parked = true;
                    return None;
                }
                // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
                FiberStep::Other(_) => {
                    *done = Some(Err(Trap::Malformed));
                    return None;
                }
            }
        }
    }

    /// Execute the current op, then stop at the next instruction whose call depth is `<= max_depth`
    /// (`None` ⇒ any depth). The shared driver for the stepping verbs — mirrors the tree-walker's
    /// `step_to_depth` (step off the current op first, then seek the next qualifying stop).
    fn step_to(&mut self, max_depth: Option<usize>, fuel: &mut u64) -> Option<super::IrPc> {
        if self.done.is_some() {
            return None;
        }
        self.stdin_parked = false;
        let Self {
            source,
            table,
            mem,
            host,
            vt,
            fibers,
            at_bp,
            done,
            op_clock,
            stdin_parked,
            funcs,
            fn_block_base,
            fn_block_types,
            debug,
            access_sink,
            scheduled_writes,
            write_cursor,
            ..
        } = self;
        *at_bp = false; // a step leaves the breakpoint-paused state
        loop {
            apply_due_writes(
                scheduled_writes,
                write_cursor,
                *op_clock,
                vt,
                source,
                mem,
                debug.as_ref(),
                fn_block_base,
                fn_block_types,
            );
            if let Some(sink) = access_sink.as_mut() {
                let cur_vm = vt.debug_active();
                emit_access(cur_vm, source, funcs, fn_block_base, *op_clock, 0, sink);
            }
            match debug_advance_fiber(vt, fibers, source, table, fuel, mem, host) {
                FiberStep::Stepped => *op_clock += 1,
                FiberStep::Finished(vals) => {
                    *op_clock += 1;
                    *done = Some(Ok(vals));
                    return None;
                }
                FiberStep::Trapped(t) => {
                    *done = Some(Err(t));
                    return None;
                }
                // Blocking-stdin park (W4): live and resumable, no clock advance (see `tick`).
                FiberStep::Other(Outcome::StdinPark) => {
                    *stdin_parked = true;
                    return None;
                }
                // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
                FiberStep::Other(_) => {
                    *done = Some(Err(Trap::Malformed));
                    return None;
                }
            }
            // Depth is *cumulative* across a coroutine boundary: a child's frames sit above the parent's
            // resume frame (`parent_depth + child stack`), so step-over of a `resume` (target =
            // parent depth) runs the child to completion, and step-out of the child body lands back in
            // the parent — while stepping *within* the child compares child-local frames as usual.
            let cur_vm = vt.debug_active();
            // Cumulative across a coroutine *or* §22-invoke boundary: the child's frames sit above the
            // parent's resume/invoke frame (`parent_depth + child stack`).
            let depth = match &vt.active_invoke {
                Some(iv) => iv.parent_depth + cur_vm.stack.len() + 1,
                None => cur_vm.stack.len() + 1,
            };
            if max_depth.is_none_or(|m| depth <= m) {
                if let Some(pc) = cur_vm.cur_ir_pc(source) {
                    return Some(pc);
                }
            }
        }
    }

    /// **Step** one instruction — descends into a call (stops at the callee's first op), the bytecode
    /// counterpart of `Inspector::step`. `None` at completion / a seam.
    pub fn step(&mut self, fuel: &mut u64) -> Option<super::IrPc> {
        self.step_to(None, fuel)
    }

    /// **Step over**: execute the current op and stop at the next op in *this* frame — running any call
    /// it makes to completion rather than descending. The counterpart of `Inspector::step_over`.
    pub fn step_over(&mut self, fuel: &mut u64) -> Option<super::IrPc> {
        let d = self.step_depth();
        self.step_to(Some(d), fuel)
    }

    /// **Step out**: run until the current function returns, stopping at the op in the caller it
    /// returned to. Runs to completion (returns `None`) when no caller frame has a remaining
    /// *steppable* op — from the outermost frame, and equally when the caller's only remaining action
    /// is its own `return` terminator: `step_to` stops only where `cur_ir_pc` is `Some`, and a
    /// terminator (`SRC_TERM`) yields `None`, so there is no op at the caller's depth to land on. The
    /// counterpart of `Inspector::step_out`; both engines agree — see the `debug_parity` pin
    /// `stepout_runs_to_completion_when_caller_immediately_returns`.
    pub fn step_out(&mut self, fuel: &mut u64) -> Option<super::IrPc> {
        let d = self.step_depth();
        self.step_to(Some(d.saturating_sub(1)), fuel)
    }

    /// Number of live call frames at the current stop (callers + the running activation) — the depth a
    /// DAP `stackTrace` would report. Inside a §14 coroutine child (step-into) this is the *child's*
    /// own frame count; see [`step_depth`](DebugRun::step_depth) for the cumulative form the stepping
    /// verbs use across the resume boundary.
    pub fn depth(&self) -> usize {
        self.reader().depth()
    }

    /// The **cumulative** call depth used by the stepping verbs: while stepping inside a coroutine child
    /// its frames count *above* the parent's resume frame (`parent_depth + child depth`), so step-over /
    /// step-out treat the resume boundary like an ordinary call. Equal to [`depth`](DebugRun::depth)
    /// when the parent itself is running.
    fn step_depth(&self) -> usize {
        let d = self.reader().depth();
        match &self.vt.active_invoke {
            Some(iv) => iv.parent_depth + d,
            None => d,
        }
    }

    /// The `IrPc` of the frame `depth` levels from the top — the bytecode counterpart of a
    /// `Inspector::backtrace` entry. `None` past the stack.
    pub fn frame_pc(&self, depth: usize) -> Option<super::IrPc> {
        self.reader().frame_pc(depth)
    }

    /// Block-local SSA value `idx` in the frame `depth` levels from the top, typed — the bytecode
    /// counterpart of `Inspector::read_ir_value`. `None` for a cross-module frame, a bad `idx`, or past
    /// the stack. A not-yet-computed slot reads as its default; the caller compares only the defined
    /// prefix (where `read_ir_value` returns `Some`).
    pub fn value_in_frame(&self, depth: usize, idx: usize) -> Option<Value> {
        self.reader().value_in_frame(depth, idx)
    }

    /// Read a **source variable by name** in the frame `depth` levels from the top — the bytecode
    /// counterpart of `Inspector::read_var`, resolving the same `VarLoc` over the §6 debug info: an
    /// `Ssa`/`SsaList` promoted scalar from the typed value slot, a `Window`/`WindowVia`/`Fixed` var
    /// from window memory. `None` if there is no debug info, the name isn't an in-scope var here, or
    /// the location can't be resolved. This is the name→value read a DAP `variables` backend needs.
    pub fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        self.reader().read_var(depth, name, width)
    }

    /// The **window address** of a source variable by name in the frame `depth` from the top — the
    /// bytecode counterpart of `Inspector::var_addr`. `Some(addr)` only for a memory-located variable
    /// (`Window`/`WindowVia`/`Fixed`); `None` for a promoted SSA scalar (no address), a name that
    /// isn't an in-scope var here, or no debug info. Feeds a DAP `variables` aggregate/array/pointer
    /// expansion (and, on the tree-walker, data breakpoints).
    pub fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        self.reader().var_addr(depth, name)
    }

    /// Read `len` bytes from the guest window at `addr` — the bytecode counterpart of
    /// `Inspector::read_window`, for a DAP `variables` backend walking an aggregate / following a
    /// pointer. Reads the active §14 coroutine child's confined window during step-into, else the
    /// parent's. Errs if the range is unmapped or the module has no memory.
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match self.mem.as_ref() {
            Some(m) => m.read_window(addr, len),
            None => Err(Trap::Malformed),
        }
    }

    /// **Write a source variable by name** (slice 8, the DAP `setVariable` backend): a promoted
    /// SSA scalar takes `value` coerced to its slot type (integers only); a memory-located var
    /// takes `value`'s low `width` bytes little-endian at its resolved window address. Refused
    /// (`false`) mid-coroutine-step, for float slots, or for an unresolvable name — fail-closed,
    /// never a guess. The DAP backend records successful writes and re-applies them at the same
    /// clock on every seek replay, so time travel stays truthful.
    pub fn write_var(&mut self, depth: usize, name: &str, value: i64, width: usize) -> bool {
        let Some(target) = self.reader().write_target(depth, name) else {
            return false;
        };
        match target {
            WriteTarget::Ssa { reg, ty } => {
                let v = match ty {
                    ValType::I32 => Value::I32(value as i32),
                    ValType::I64 => Value::I64(value),
                    _ => return false,
                };
                match self.vt.active.regs.get_mut(reg) {
                    Some(r) => {
                        *r = Reg::from_value(v);
                        true
                    }
                    None => false,
                }
            }
            WriteTarget::Win { addr } => {
                let w = width.clamp(1, 8);
                self.write_window(addr, &value.to_le_bytes()[..w])
            }
        }
    }

    /// **Write bytes into the guest window** (slice 8, the DAP `writeMemory` backend). `false` if
    /// the range is unmapped or the module has no memory.
    pub fn write_window(&mut self, addr: u64, bytes: &[u8]) -> bool {
        self.mem
            .as_mut()
            .and_then(|m| m.write_bytes(addr, bytes))
            .is_some()
    }

    /// The running frame's block-local SSA value `idx` ([`value_in_frame`] at depth 0).
    pub fn value(&self, idx: usize) -> Option<Value> {
        self.value_in_frame(0, idx)
    }

    /// The run result once finished (`None` while still running).
    pub fn result(&self) -> Option<&Result<Vec<Value>, Trap>> {
        self.done.as_ref()
    }

    /// The session's powerbox host (`Host::new_with_host`'s grant), for reading effects a debugged guest
    /// produced — captured stdout/stderr, and the [`CapTape`](Host::cap_tape) a reverse `seek` replays so
    /// a **powerbox** run (streams/clock/exit) re-executes with identical cap inputs.
    pub fn host(&self) -> &Host {
        &self.host
    }
    /// Mutable powerbox host — e.g. to drain captured stdout between stops.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }
}

/// Whether `m` can spawn a second vCPU — it contains a `thread.spawn` op somewhere. The DAP backend
/// routes such a module to the multithreaded [`ScheduledDebugRun`] instead of the single-vCPU
/// [`DebugRun`]; a spawn-free module stays on the (reverse- and watch-capable) single-vCPU path.
pub fn module_spawns_threads(m: &Module) -> bool {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .any(|i| matches!(i, Inst::ThreadSpawn { .. }))
}

/// The outcome of one [`ScheduledDebugRun`] pump — the multi-vCPU counterpart of a `DebugRun` stop.
#[derive(Debug)]
pub enum SchedStop {
    /// A stop fired in some thread; that thread is now the stopped + focused one
    /// ([`stopped_task`](ScheduledDebugRun::stopped_task)). `reason` says why.
    Break { pc: super::IrPc, reason: SchedBreak },
    /// The root vCPU finished — the run's result (or trap).
    Finished(Result<Vec<Value>, Trap>),
    /// No thread is runnable and the root hasn't finished: a `memory.wait`/deadlock the debug
    /// scheduler can't advance (it drives only `thread.spawn`/`join`).
    Blocked,
    /// A thread reached an op outside the debug scheduler's subset — only JIT tier-up (never enabled on
    /// this engine). Threads, `wait`/`notify`, fibers, `instantiate`/`instantiate_module`, and §14
    /// coroutines (step-into, with the coroutine's vCPU pinned across the body) are all handled.
    Declined,
}

/// Why a [`SchedStop::Break`] fired — mapped to the DAP stop reason by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedBreak {
    /// A pc in the run-shared breakpoint set (in whichever thread reached it).
    Breakpoint,
    /// A window watchpoint: the about-to-run op touches `[addr, addr+len)` with a matching kind. The
    /// stop is *before* the access applies.
    Watchpoint { addr: u64, write: bool },
    /// A single-step / step-over / step-out landed the stepping thread at its target.
    Step,
}

/// One scheduled vCPU under the multi-vCPU debugger. `Clone` for time-travel checkpointing (W1): a
/// scheduled checkpoint captures each task's state so a reverse `seek` restarts from the nearest
/// snapshot instead of turn 0.
#[derive(Clone)]
enum DbgTaskState {
    Runnable,
    /// Parked on `thread.join` of task `child` (handle `slot`); its result lands at `dst` on wake.
    BlockedJoin {
        child: usize,
        slot: usize,
        dst: u32,
    },
    /// Parked on `memory.wait` at futex key `key` until `memory.notify` or the logical `clock` reaches
    /// `deadline`; the status (`WAIT_WOKEN` / `WAIT_TIMED_OUT`) lands at `dst`.
    BlockedWait {
        key: u64,
        deadline: u64,
        dst: u32,
    },
    /// Finished — result (or trap) retained for a joiner.
    Done(Result<Vec<Value>, Trap>),
}

struct DbgTask {
    /// The reified continuation of this vCPU: its active `Vm` plus its §12 fiber resume `chain` (a
    /// `cont.resume` switches `vt.active` into a fiber; `suspend` / a fiber return switches back).
    vt: VTask,
    /// This vCPU's `thread.spawn` / `instantiate` children (handle = index → global task index; `None`
    /// = joined). Both seams share one handle namespace, so `Instantiator.join` (→ `ThreadJoin`) joins
    /// an `instantiate` child through the same machinery.
    threads: Vec<Option<usize>>,
    /// The runtime environment this vCPU steps against: `None` = the shared domain (root + its
    /// `thread.spawn` siblings, over `self.mem`/`self.host`/`self.table`); `Some(k)` = the confined
    /// [`extra_envs`](ScheduledDebugRun::extra_envs)`[k]` of a §14 `instantiate` child. The debug-engine
    /// counterpart of the production [`TaskSlot::env`].
    env: Option<usize>,
    state: DbgTaskState,
    /// Paused on a just-reported breakpoint — step one op past it before the next scan makes progress
    /// (the per-task analogue of [`DebugRun::at_bp`], so a loop-body breakpoint re-fires each iteration).
    at_bp: bool,
}

/// A §14 `instantiate` **confined executor child**'s runtime under the multi-vCPU debug scheduler — the
/// debug-engine counterpart of the production [`ChildEnv`]. Its `mem` is a `nested_view` sub-window
/// sharing the parent's backing (the confinement masking is the production primitive, unchanged — the
/// debugger only drives an already-confined child op-by-op), `host` an attenuated powerbox (an
/// `Instantiator` + `AddressSpace`, each over `[0, child_size)`), `table` a fresh natural dispatch table
/// over module 0 (no installed §22 units), and `fuel` a sub-allocated quota. Plain `Host` (the debug
/// scheduler is single-threaded and the §3.6 live-call/serve machinery is not yet driven here).
struct DbgEnv {
    mem: Option<Mem>,
    host: Host,
    table: SharedSlots,
    fuel: u64,
}

/// One scheduled vCPU's captured state inside a [`ScheduledSnapshot`] — its full `VTask` continuation
/// (active `Vm`, active fiber id, resume `chain`, same-module coroutine children, active-coroutine
/// cursor, durable shadow-SP), the join-handle table, the env index (`Some(k)` for a §14 `instantiate`
/// child — see [`ScheduledSnapshot::extra_envs`]), the run state, and the breakpoint-skip flag. The
/// fiber `Vm`s share the run's window(s), and each coroutine's window is a `nested_view` sharing the
/// parent backing, so their bytes ride in the snapshot's window bytes; `restore` rebuilds the `VTask`.
struct DbgTaskSnapshot {
    active: Vm,
    active_id: usize,
    chain: Vec<(usize, Vm, u32)>,
    root_shadow_sp: u64,
    threads: Vec<Option<usize>>,
    env: Option<usize>,
    state: DbgTaskState,
    at_bp: bool,
}

/// A multi-vCPU time-travel **checkpoint** (DEBUGGING.md W1): the re-executable state of a
/// [`ScheduledDebugRun`] at global [`turn`](ScheduledSnapshot::turn), so a reverse `seek`/`step_back`
/// restarts a replay here instead of from turn 0. Captured only for the simple threaded subset (see
/// [`ScheduledDebugRun::checkpointable`]: no §12 fibers, no §14 coroutines/`instantiate` children, a
/// pristine shared window, a restorable host) — where the per-task active `Vm`s + the shared window
/// bytes + the host substate + the scheduler clocks fully determine the continuation. The scheduled
/// counterpart of [`DebugRunSnapshot`]. Opaque to the DAP backend, which stores it in a ladder and hands
/// it back to [`ScheduledDebugRun::restore`].
pub struct ScheduledSnapshot {
    turn: u64,
    clock: u64,
    tasks: Vec<DbgTaskSnapshot>,
    /// The **run-shared** §12 fiber registry (one handle namespace across all vCPUs — a fiber migrates,
    /// D57), reconstructed verbatim on restore. The parked fiber `Vm`s share the run's window.
    fibers: Vec<FiberState>,
    /// The §14 `instantiate`-child environments (handle = index; a task's [`DbgTask::env`] indexes this).
    /// See [`EnvSnapshot`].
    extra_envs: Vec<EnvSnapshot>,
    /// The non-primary [`ModuleSource`] units (a §14 **separate-module** `instantiate_module` child's or
    /// coroutine's pushed program), re-pushed on restore so a `module >= 1` frame resolves. Empty for a
    /// same-module-only run. Cheap `Arc` clones — the compiled units are immutable.
    extra_units: Vec<std::sync::Arc<Compiled>>,
    /// The run window's full memory state — committed bytes **and** page-protection map
    /// ([`Mem::layout_snapshot`]), reinstated via [`Mem::restore_layout`] on restore. All tasks share this
    /// one window; capturing its protection map admits a **page-mapping** run (`map`/`unmap`/`protect`/grow).
    mem: Option<super::MemLayout>,
    host: super::HostReplaySubstate,
}

/// A §14 `instantiate`-child environment ([`DbgEnv`]) inside a [`ScheduledSnapshot`]. Like a coroutine
/// child, its window is a `nested_view` sharing the root backing region — so its bytes ride in the
/// snapshot's window bytes and only the view geometry (`win_base`/`size_log2`) is stored; `restore`
/// rebuilds the view over the reseeded shared window. Its attenuated powerbox (an `Instantiator` +
/// `AddressSpace`, each over `[0, child_size)`) is deterministic in `child_size = 1 << size_log2`, so
/// only the host replay substate is carried; a natural dispatch table over the child's [`module`] (a
/// **separate-module** `instantiate_module` child's pushed unit rides in `extra_units`) and the
/// sub-allocated `fuel` complete it. The child's own page-protection map (`prot`) is captured alongside,
/// admitting a child that `map`/`unmap`/`protect`ed its own window; its bytes still ride in the shared
/// snapshot. A §13 region-aliased child stays outside the checkpointable subset.
struct EnvSnapshot {
    win_base: u64,
    size_log2: u8,
    /// The child's module in the shared source (`0` = same-module `instantiate`; `>= 1` = a
    /// **separate-module** `instantiate_module` child, whose pushed unit is captured in the run
    /// snapshot's `extra_units`). Its natural dispatch table is rebuilt over this index.
    module: usize,
    host: super::HostReplaySubstate,
    fuel: u64,
    /// The child's own page-protection map ([`Mem::prot_snapshot`]), reinstalled with
    /// [`Mem::install_prot`] on restore (its bytes ride in the shared snapshot). Empty for a pristine child.
    prot: Vec<(u64, super::PageProt)>,
}

impl ScheduledSnapshot {
    /// The global turn this checkpoint was taken at — the ladder key the backend searches.
    pub fn turn(&self) -> u64 {
        self.turn
    }
}

/// A **multi-vCPU** debug session on the bytecode engine (DEBUGGING.md Milestone B, bytecode side): a
/// deterministic cooperative debug scheduler over one shared `Mem` for a `thread.spawn`/`join` guest.
/// Mirrors the tree-walker's [`Inspector::attach_scheduled`](crate::Inspector) — a run-shared breakpoint
/// set fires in **whichever** vCPU reaches it (stopping *before* the op), `stopped_task` reports which,
/// and `select_task` focuses read-inspection (backtrace / `read_var` / `read_window`) on any live thread
/// while stopped in another. The schedule is a reproducible lowest-index-runnable, one-op-per-turn pick
/// (the debuggable analogue of the production `drive`), so the interleaving is deterministic — which is
/// what makes **reverse debugging** (`tick`-replay to a global `turn`) and **cross-thread watchpoints**
/// (the per-op seam checks the armed ranges in whichever thread) sound. Stepping is depth-aware
/// (in/over/out), **`memory.wait`/`notify`** park/wake threads (a stuck set advances a logical `clock`
/// to the earliest wait deadline, exactly as the production `drive`), and **§12 fibers** switch each
/// vCPU's active continuation (breakpoints fire inside a resumed fiber; the fiber registry is run-shared
/// so a fiber migrates across vCPUs — D57), and **§14 `instantiate` / `instantiate_module`** spawn a
/// confined executor child as its own scheduled vCPU (its own [`DbgEnv`] — window / powerbox / quota —
/// joinable through the shared thread machinery; a separate-module child runs its own pushed module,
/// and nesting composes to any depth), and **§14 coroutines** are stepped op-by-op (step-into) with the
/// coroutine's vCPU **pinned** across the body so a `resume` stays atomic w.r.t. other vCPUs. The only
/// op outside the subset (→ [`SchedStop::Declined`]) is JIT tier-up (never enabled here).
pub struct ScheduledDebugRun {
    source: std::sync::Arc<ModuleSource>,
    table: SharedSlots,
    mem: Option<Mem>,
    host: Host,
    tasks: Vec<DbgTask>,
    /// §14 `instantiate` confined children's environments (handle = index; a task's [`DbgTask::env`] is
    /// `Some(k)` into this). Grown as children spawn; rebuilt deterministically on a reverse-`seek`
    /// replay. Not torn down mid-run (a finished child's env is inert — revocation semantics deferred).
    extra_envs: Vec<DbgEnv>,
    /// The **run-shared** §12 fiber registry (one handle namespace across all vCPUs; a fiber created on
    /// one can be resumed on another — D57). Rebuilt deterministically on a reverse `seek` replay.
    fibers: Vec<FiberState>,
    fn_block_base: Vec<Vec<u32>>,
    fn_block_types: Vec<Vec<Vec<ValType>>>,
    debug: Option<DebugInfo>,
    /// The IR functions, for computing the effective address of the op about to run when a watchpoint
    /// is armed (`watch_hit_before`). `Arc` so a reverse `seek` rebuild is cheap.
    funcs: std::sync::Arc<[Func]>,
    breakpoints: Vec<super::IrPc>,
    /// Run-shared window watchpoints (DEBUGGING.md W2, cross-thread): `(addr, len, kind)`. Empty in the
    /// common case, so the per-op `access_of` computation is skipped entirely.
    watchpoints: Vec<(u64, u64, super::WatchKind)>,
    /// The session's optional per-op access sink ([`AccessSinkFn`]) — fired before every module-0
    /// op with the global `turn` and the **executing task index** (the vCPU attribution host-side
    /// models key on). `None` (the default) is zero-cost; the DAP backend re-installs it on every
    /// `seek` rebuild, like watchpoints.
    access_sink: Option<AccessSinkFn>,
    /// The optional **scheduler trace tape** ([`SchedTraceEvent`], slice 6) — armed by
    /// [`set_sched_trace`](ScheduledDebugRun::set_sched_trace); `None` (the default) is zero-cost.
    /// Re-armed by the DAP backend on `seek` rebuilds (the replay refills it deterministically).
    sched_trace: Option<Vec<SchedTraceEvent>>,
    /// Scheduled debugger writes ([`ScheduledWrite`], slice 8) + the next-un-applied cursor — the
    /// scheduled-engine twin of `DebugRun::scheduled_writes`.
    scheduled_writes: Vec<(u64, ScheduledWrite)>,
    write_cursor: usize,
    /// The **seeded pick** (slice 7): `Some(seed)` chooses uniformly among the runnable set via
    /// `splitmix64(seed ^ turn)` — an adversarial-variation knob whose choice is a pure function
    /// of `(seed, turn)`, so replay reproduces it with no captured scheduler state. `None` (the
    /// default) keeps the original lowest-index pick.
    sched_seed: Option<u64>,
    /// Recorded **forced switches** (slice 7): concrete `(turn, task)` overrides, resolved at
    /// record time and re-applied by the DAP backend on rebuilds — so a `seek` replays them at the
    /// identical turns. Empty (the default) is zero-cost.
    forced: Vec<(u64, usize)>,
    /// Set when `drive` stopped *before* an op that hits a watchpoint (the access hasn't applied yet);
    /// taken by the backend to report `StopReason::Watchpoint`.
    last_watch: Option<(u64, bool)>,
    /// The task index paused on a breakpoint (stepping drives it); `None` while running.
    stopped: Option<usize>,
    /// The task `select_task` focuses read-inspection on; reset to the stopped thread on each stop.
    focus: usize,
    /// Global count of visible ops executed across all vCPUs — the scheduled-mode logical clock and the
    /// reverse-`seek` coordinate.
    turn: u64,
    /// The `memory.wait` deadline clock (advanced only when the whole run is stuck-waiting, to the
    /// earliest deadline). Separate from `turn`: it measures futex timeout time, not ops.
    clock: u64,
}

/// Mark task `ti` done and wake any joiner parked on it (delivering its result / propagating a trap) —
/// the debug-scheduler counterpart of the production [`complete`].
/// One record on the **scheduler trace tape** (INTERACTIVE_EMBEDDING.md slice 6): the cooperative
/// debug scheduler's own decisions — turns, parks, wakes with both identities, spawns —
/// observation-only, derived by diffing task states across each decision (no scheduling logic is
/// touched; invariant 4 holds — the host records, never chooses differently). `turn` is the global
/// turn at the decision. The tape is deterministic: the same run replayed yields the identical
/// tape (the schedule itself is deterministic), which is what makes it a sound timeline source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchedTraceEvent {
    /// `task` ran the op at global `turn`.
    Turn { turn: u64, task: usize },
    /// `task` parked joining `child` (`thread.join` on a live child).
    ParkJoin {
        turn: u64,
        task: usize,
        child: usize,
    },
    /// `task` parked on the futex word at window offset `key` (`memory.wait`).
    ParkWait { turn: u64, task: usize, key: u64 },
    /// `waker`'s `memory.notify` woke `wakee` off its futex wait.
    WakeNotify {
        turn: u64,
        waker: usize,
        wakee: usize,
    },
    /// `waker`'s completion (thread exit) woke joiner `wakee`.
    WakeJoin {
        turn: u64,
        waker: usize,
        wakee: usize,
    },
    /// `task` woke from a timed `memory.wait` whose deadline passed (the picker's clock advance).
    WakeTimeout { turn: u64, task: usize },
    /// `parent`'s `thread.spawn` created `task`.
    Spawn {
        turn: u64,
        parent: usize,
        task: usize,
    },
}

/// A compact `(state-tag, aux)` per task for the trace differ: 0 = runnable, 1 = blocked-join
/// (aux = child), 2 = blocked-wait (aux = key), 3 = done.
fn trace_tags(tasks: &[DbgTask]) -> Vec<(u8, u64)> {
    tasks
        .iter()
        .map(|t| match t.state {
            DbgTaskState::Runnable => (0, 0),
            DbgTaskState::BlockedJoin { child, .. } => (1, child as u64),
            DbgTaskState::BlockedWait { key, .. } => (2, key),
            DbgTaskState::Done(_) => (3, 0),
        })
        .collect()
}

/// Diff task states across one advance by `actor` at `turn`, appending the park/wake/spawn events
/// the transition implies. A task beyond `before`'s length is a fresh spawn by the actor.
fn trace_diff(
    before: &[(u8, u64)],
    tasks: &[DbgTask],
    turn: u64,
    actor: usize,
    out: &mut Vec<SchedTraceEvent>,
) {
    let now = trace_tags(tasks);
    for (j, &(nt, naux)) in now.iter().enumerate() {
        match before.get(j) {
            None => out.push(SchedTraceEvent::Spawn {
                turn,
                parent: actor,
                task: j,
            }),
            Some(&(wt, waux)) if wt == nt && waux == naux => {}
            Some(&(wt, _)) => match (wt, nt) {
                (_, 1) => out.push(SchedTraceEvent::ParkJoin {
                    turn,
                    task: j,
                    child: naux as usize,
                }),
                (_, 2) => out.push(SchedTraceEvent::ParkWait {
                    turn,
                    task: j,
                    key: naux,
                }),
                (2, 0) => out.push(SchedTraceEvent::WakeNotify {
                    turn,
                    waker: actor,
                    wakee: j,
                }),
                (1, 0) => out.push(SchedTraceEvent::WakeJoin {
                    turn,
                    waker: actor,
                    wakee: j,
                }),
                _ => {} // a completion or re-key — no timeline edge
            },
        }
    }
}

/// The pick-phase differ: the only transition a pick can cause is a timed-out `memory.wait` waking
/// (`dbg_pick_runnable`'s clock advance), so any blocked-wait → runnable here is a `WakeTimeout`.
fn trace_pick_diff(
    before: &[(u8, u64)],
    tasks: &[DbgTask],
    turn: u64,
    out: &mut Vec<SchedTraceEvent>,
) {
    let now = trace_tags(tasks);
    for (j, &(nt, _)) in now.iter().enumerate() {
        if let Some(&(2, _)) = before.get(j) {
            if nt == 0 {
                out.push(SchedTraceEvent::WakeTimeout { turn, task: j });
            }
        }
    }
}

fn dbg_complete(tasks: &mut [DbgTask], ti: usize, res: Result<Vec<Value>, Trap>) {
    let mut work = vec![(ti, res)];
    while let Some((done, res)) = work.pop() {
        tasks[done].state = DbgTaskState::Done(res.clone());
        for (j, t) in tasks.iter_mut().enumerate() {
            let DbgTaskState::BlockedJoin { child, slot, dst } = t.state else {
                continue;
            };
            if child != done {
                continue;
            }
            t.threads[slot] = None;
            match &res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    t.vt.active.set(dst, Reg::from_value(v));
                    t.state = DbgTaskState::Runnable;
                }
                Err(trap) => work.push((j, Err(trap.clone()))),
            }
        }
    }
}

/// `thread.spawn`: add a child vCPU running `func(sp, arg)` (sharing the domain), write its handle to
/// the spawner's `dst`. Mirrors the production `drive`'s `Spawn` arm for the debuggable subset.
#[allow(clippy::too_many_arguments)]
fn dbg_spawn(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    func: u32,
    sp: i64,
    arg: i64,
    dst: u32,
    module: usize,
    source: &ModuleSource,
) -> Result<(), Trap> {
    // Module-aware, as the drive arms: `func` is the spawning frame's module's index.
    let cm = source.get(module).ok_or(Trap::Malformed)?;
    if func as usize >= cm.progs.len() {
        return Err(Trap::Malformed);
    }
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // thread bomb
    }
    let mut vt = VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
    vt.active.module = module;
    vt.active.home = module;
    let env = tasks[ti].env; // a thread inherits its spawner's environment (shares its window)
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt,
        threads: Vec::new(),
        env,
        state: DbgTaskState::Runnable,
        at_bp: false,
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// The per-op driver for a scheduled debug task, selecting its runtime context: the shared domain
/// (`env = None`) or its confined [`DbgEnv`] (`env = Some(k)`). Centralizes the split borrow so the
/// unified `drive`/`tick` pumps stay a single call. Mirrors the production `drive`'s `RunCtx` selection.
#[allow(clippy::too_many_arguments)]
fn dbg_advance_task(
    tasks: &mut [DbgTask],
    ti: usize,
    extra_envs: &mut [DbgEnv],
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    match tasks[ti].env {
        None => debug_advance_fiber(&mut tasks[ti].vt, fibers, source, table, fuel, mem, host),
        Some(k) => {
            let e = &mut extra_envs[k];
            debug_advance_fiber(
                &mut tasks[ti].vt,
                fibers,
                source,
                &e.table,
                &mut e.fuel,
                &mut e.mem,
                &mut e.host,
            )
        }
    }
}

/// Outcome of [`service_advance`]: `Ran` = a step this engine services (op / thread spawn·join /
/// futex wait·notify / §14 instantiate) was dispatched and `turn` ticked; `Declined` = a coroutine
/// or tier-up op this scheduled engine does not drive (`turn` left untouched, caller bails its own
/// way).
enum Serviced {
    Ran,
    Declined,
}

/// Advance task `ti` one step and service the scheduler seam it produced. This is the shared core of
/// [`ScheduledDebugRun::drive`] and [`ScheduledDebugRun::tick`]: both dispatch the *same* set of
/// [`Outcome`]s with the *same* rejections, so a new seam is added here **once** rather than in two
/// places — a `tick` (the reverse-`seek` replay path) that silently declined what `drive` services
/// would desync replay from live runs (DEBUGGING.md; INVARIANTS #9 observability corollary). `turn`
/// is ticked for every serviced step, matching each engine's per-op clock; on `Declined` it is left
/// untouched so each caller keeps its own bail (`drive` → `SchedStop::Declined` with no tick; `tick`
/// → tick its clock and stop the replay).
#[allow(clippy::too_many_arguments)]
fn service_advance(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    clock: u64,
    turn: &mut u64,
) -> Serviced {
    let step_res = dbg_advance_task(
        tasks, ti, extra_envs, fibers, source, table, fuel, mem, host,
    );
    tasks[ti].at_bp = false;
    match step_res {
        // A fiber switch (or a plain op) — the vCPU advanced one op, stays runnable.
        FiberStep::Stepped => *turn += 1,
        FiberStep::Finished(vals) => {
            *turn += 1;
            dbg_complete(tasks, ti, Ok(vals));
        }
        FiberStep::Trapped(t) => {
            *turn += 1;
            dbg_complete(tasks, ti, Err(t));
        }
        // A scheduler seam: the ones this engine dispatches, else `Declined`.
        FiberStep::Other(outcome) => match outcome {
            Outcome::ThreadSpawn {
                func,
                sp,
                arg,
                dst,
                module,
            } => {
                *turn += 1;
                if let Err(t) = dbg_spawn(tasks, ti, func, sp, arg, dst, module, source) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            Outcome::ThreadJoin { handle, dst } => {
                *turn += 1;
                dbg_join(tasks, ti, handle, dst);
            }
            Outcome::MemoryWait {
                base,
                expected,
                width,
                timeout,
                dst,
            } => {
                *turn += 1;
                dbg_wait(tasks, ti, mem, clock, base, expected, width, timeout, dst);
            }
            Outcome::MemoryNotify { base, count, dst } => {
                *turn += 1;
                dbg_notify(tasks, ti, base, count, dst);
            }
            // §14 `instantiate` (op 0): spawn a confined executor child as its own scheduled vCPU.
            Outcome::Instantiate {
                ibase,
                isize: isz,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            } => {
                *turn += 1;
                // op 11 (named-grant spawn) / a §3d budget record is not driven by the debugger path.
                if grants.is_some() || budget != 0 {
                    dbg_complete(tasks, ti, Err(Trap::Malformed));
                } else if let Err(t) = dbg_instantiate(
                    tasks, ti, extra_envs, source, mem, *fuel, ibase, isz, entry, off, size_log2,
                    quota, dst,
                ) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            // §14 `instantiate_module` (op 5): a confined child running a granted separate module.
            Outcome::InstantiateModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            } => {
                *turn += 1;
                // op 13 (named-grant spawn) / a §3d budget record is not driven by the debugger path.
                if grants.is_some() || budget != 0 {
                    dbg_complete(tasks, ti, Err(Trap::Malformed));
                } else if let Err(t) = dbg_instantiate_module(
                    tasks, ti, extra_envs, source, mem, *fuel, host, mh, ibase, isz, entry, off,
                    size_log2, quota, dst,
                ) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            // coroutine / tier-up — outside this engine's slice.
            _ => return Serviced::Declined,
        },
    }
    Serviced::Ran
}

/// §14 `instantiate` (op 0) under the debug scheduler: build a **confined executor child** as a new
/// scheduled vCPU with its own [`DbgEnv`] (window / attenuated powerbox / quota), registered as a
/// child handle of `ti` (so `Instantiator.join` → `ThreadJoin` joins it). The debug-engine counterpart
/// of the production `drive`'s `Instantiate` arm — same carve + attenuation + natural table. Writes the
/// handle (or `EINVAL`) to `dst`; `Err(ThreadFault)` on the vCPU-count bomb (the caller completes `ti`).
#[allow(clippy::too_many_arguments)]
fn dbg_instantiate(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    source: &ModuleSource,
    shared_mem: &Option<Mem>,
    shared_fuel: u64,
    ibase: u64,
    isz: u64,
    entry: i64,
    off: i64,
    size_log2: i64,
    quota: i64,
    dst: u32,
) -> Result<(), Trap> {
    let c0 = source.primary();
    // A confined child's entry is `(i64 instantiator) -> (i64)` or `(i64 instantiator, i64 address_space)
    // -> (i64)`; the latter also gets an `AddressSpace` grant so it manages its own pages.
    let sig = c0.sigs.get(entry as u64 as usize);
    let want_as = sig.is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
    let ok_entry = sig.is_some_and(|(p, r)| child_entry_ok(p, r));
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off_u = off as u64;
    let fits = carve_fits(
        off_u,
        size_log2,
        isz,
        ibase,
        shared_mem.as_ref().map_or(0, |m| m.null_guard),
    );
    if !ok_entry || !fits {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::EINVAL as i32));
        return Ok(());
    }
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // instantiate bomb
    }
    // Holder-relative `ibase`/`off` → backing-absolute base (so nesting composes); parent window base
    // and fuel come from the parent's environment (shared or its own confined env).
    let (pbase, pfuel) = match tasks[ti].env {
        None => (
            shared_mem.as_ref().map_or(0, |m| m.window.base()),
            shared_fuel,
        ),
        Some(k) => (
            extra_envs[k].mem.as_ref().map_or(0, |m| m.window.base()),
            extra_envs[k].fuel,
        ),
    };
    let abs_base = pbase + ibase + off_u;
    let child_mem = match tasks[ti].env {
        None => shared_mem
            .as_ref()
            .map(|m| m.nested_view(abs_base, size_log2 as u8)),
        Some(k) => extra_envs[k]
            .mem
            .as_ref()
            .map(|m| m.nested_view(abs_base, size_log2 as u8)),
    };
    // Attenuated powerbox over the child's *own* `[0, child_size)`: an `Instantiator` (so it can nest —
    // confinement composes) and an `AddressSpace`; these are its entry arguments.
    let mut child_host = Host::new();
    let cinst = child_host.grant_instantiator(0, child_size);
    let cas = child_host.grant_address_space(0, child_size);
    let child_args = if want_as {
        vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
    } else {
        vec![Value::I64(cinst as i64)]
    };
    let child_fuel = if quota <= 0 {
        pfuel
    } else {
        (quota as u64).min(pfuel)
    };
    // The child is its own domain: a fresh natural table over module 0 (no installed §22 units).
    let child_table = build_table(c0.progs.len(), 0);
    let child_vt = VTask::new(&c0, entry as u64 as usize, &child_args)?;
    let eidx = extra_envs.len();
    extra_envs.push(DbgEnv {
        mem: child_mem,
        host: child_host,
        table: child_table,
        fuel: child_fuel,
    });
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt: child_vt,
        threads: Vec::new(),
        env: Some(eidx),
        state: DbgTaskState::Runnable,
        at_bp: false,
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// §14 `instantiate_module` (op 5) under the debug scheduler: like [`dbg_instantiate`], but the confined
/// executor child runs a **host-granted separate `Module`** — resolve it from the powerbox, compile it,
/// **push it to the shared source** (so it dispatches by its own index, like a separate-module coroutine,
/// slice 14c), materialize its data segments into the carve, and run the child over its own module. The
/// debug-engine counterpart of the production `drive`'s `InstantiateModule` arm. Writes the handle (or
/// `EINVAL`) to `dst`; `Err` for a forged/closed module handle, an un-lowerable module, or the vCPU bomb.
#[allow(clippy::too_many_arguments)]
fn dbg_instantiate_module(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    source: &ModuleSource,
    shared_mem: &Option<Mem>,
    shared_fuel: u64,
    host: &Host,
    mh: i32,
    ibase: u64,
    isz: u64,
    entry: i64,
    off: i64,
    size_log2: i64,
    quota: i64,
    dst: u32,
) -> Result<(), Trap> {
    // Resolve + clone the granted module from the powerbox (mirrors production: the module handle is
    // resolved against the shared host). A forged/closed/wrong-type handle is an inert CapFault.
    let (cfuncs, cmem_log2, cdata) = {
        let g = host.resolve_module(mh)?;
        (g.funcs.clone(), g.memory_log2, g.data.clone())
    };
    let child_compiled = match compile_module(&cfuncs) {
        Some(c) => c,
        None => return Err(Trap::Malformed),
    };
    // Entry sig is validated against the *child module*; a separate-module child's carve must equal its
    // declared memory (§14 transparency — it runs exactly as it would standalone).
    let sig = child_compiled.sigs.get(entry as u64 as usize);
    let want_as = sig.is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
    let ok_entry = sig.is_some_and(|(p, r)| child_entry_ok(p, r));
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off_u = off as u64;
    let fits = carve_fits(
        off_u,
        size_log2,
        isz,
        ibase,
        shared_mem.as_ref().map_or(0, |m| m.null_guard),
    );
    let mod_ok = cmem_log2 == Some(size_log2 as u8);
    if !ok_entry || !fits || !mod_ok {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::EINVAL as i32));
        return Ok(());
    }
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault);
    }
    let (pbase, pfuel) = match tasks[ti].env {
        None => (
            shared_mem.as_ref().map_or(0, |m| m.window.base()),
            shared_fuel,
        ),
        Some(k) => (
            extra_envs[k].mem.as_ref().map_or(0, |m| m.window.base()),
            extra_envs[k].fuel,
        ),
    };
    let abs_base = pbase + ibase + off_u;
    // Materialize the module's data segments into the carve before the child runs, then the view.
    let child_mem = {
        let pm: Option<&Mem> = match tasks[ti].env {
            None => shared_mem.as_ref(),
            Some(k) => extra_envs[k].mem.as_ref(),
        };
        if let Some(m) = pm {
            for d in cdata.iter() {
                if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
                    for (k, &b) in d.bytes.iter().enumerate() {
                        m.set_byte(abs_base + d.offset + k as u64, b);
                    }
                }
            }
        }
        pm.map(|m| m.nested_view(abs_base, size_log2 as u8))
    };
    let mut child_host = Host::new();
    let cinst = child_host.grant_instantiator(0, child_size);
    let cas = child_host.grant_address_space(0, child_size);
    let child_args = if want_as {
        vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
    } else {
        vec![Value::I64(cinst as i64)]
    };
    let child_fuel = if quota <= 0 {
        pfuel
    } else {
        (quota as u64).min(pfuel)
    };
    // Push the child's compiled module and run the child over it — its own domain: a natural table
    // mapping into *its* pushed module index (the mutable-Domain step, like a separate-module coroutine).
    let progs_len = child_compiled.progs.len();
    let cm = source.push(child_compiled);
    let child_table = build_table_for(progs_len, 0, cm as u32);
    let cunit = source.get(cm).ok_or(Trap::Malformed)?;
    let mut child_vt = VTask::new(&cunit, entry as u64 as usize, &child_args)?;
    child_vt.active.module = cm;
    child_vt.active.home = cm;
    let eidx = extra_envs.len();
    extra_envs.push(DbgEnv {
        mem: child_mem,
        host: child_host,
        table: child_table,
        fuel: child_fuel,
    });
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt: child_vt,
        threads: Vec::new(),
        env: Some(eidx),
        state: DbgTaskState::Runnable,
        at_bp: false,
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// `thread.join`: deliver a finished child's result now, else park the joiner. Mirrors `drive`'s `Join`.
fn dbg_join(tasks: &mut [DbgTask], ti: usize, handle: i32, dst: u32) {
    let slot = match super::resolve_thread(&tasks[ti].threads, handle) {
        Ok(s) => s,
        Err(t) => {
            dbg_complete(tasks, ti, Err(t));
            return;
        }
    };
    let child = tasks[ti].threads[slot].expect("resolve_thread checked liveness");
    match &tasks[child].state {
        DbgTaskState::Done(res) => {
            let res = res.clone();
            tasks[ti].threads[slot] = None;
            match res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    tasks[ti].vt.active.set(dst, Reg::from_value(v));
                }
                Err(t) => dbg_complete(tasks, ti, Err(t)),
            }
        }
        _ => tasks[ti].state = DbgTaskState::BlockedJoin { child, slot, dst },
    }
}

/// §22 `Jit.install` (op 3) under the debug engine: resolve authority + the unit's funcs from the host
/// (a forged/cross-domain handle is an inert `CapFault` → trap), compile the unit to bytecode, and
/// install it into the debug run's shared `(source, table)` — the debug-engine counterpart of the
/// production `drive`'s `JitInstall` arm. Serviced inline in [`debug_advance_fiber`] (it mutates only
/// `vt.active` + the shared table, spawning no scheduler task), so both the single-vCPU `DebugRun` and
/// the `ScheduledDebugRun` reach it. Writes the slot (or `-ENOSPC`, an ordinary value) to `dst`; `Err`
/// traps the vCPU (`CapFault` forged handle, `Malformed` unit outside bytecode coverage — the one place
/// a guest-provided unit can outrun coverage, with no tree-walker fallback mid-run).
fn dbg_jit_install(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    table: &SharedSlots,
    h: i32,
    code: i32,
    dst: u32,
) -> Result<(), Trap> {
    let funcs = host.resolve_jit_domain(h).and_then(|domain| {
        let (cd, cu) = host.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
    })?;
    let res = match compile_module(&funcs) {
        Some(unit) => match jit_install_into(source, table, unit) {
            Some(slot) => slot as i64,
            None => super::ENOSPC,
        },
        None => return Err(Trap::Malformed), // unit op outside coverage
    };
    vt.active.set(dst, Reg::from_i64(res));
    Ok(())
}

/// §22 `Jit.uninstall` (op 4) under the debug engine: authority-check the domain handle, then clear the
/// installed table slot (`0`/`-EINVAL` to `dst`). Mirrors `drive`'s `JitUninstall` arm; serviced inline
/// in [`debug_advance_fiber`].
fn dbg_jit_uninstall(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    table: &SharedSlots,
    h: i32,
    slot: i64,
    dst: u32,
) -> Result<(), Trap> {
    host.resolve_jit_domain(h)?; // authority (forged handle → CapFault)
    let n_real = source.primary().progs.len();
    let res = if jit_uninstall_from(source, table, slot as usize, n_real) {
        0
    } else {
        super::EINVAL
    };
    vt.active.set(dst, Reg::from_i64(res));
    Ok(())
}

/// Shared prep for §22 `Jit.invoke` on the debug engine (the leaf and step-into paths both use it):
/// resolve authority + the unit's funcs from the host (forged/cross-domain → `CapFault`), compile the
/// unit (out-of-coverage → `Malformed`), arity-check its entry (func 0) against the call's
/// (code-stripped) signature (`CapFault` on mismatch), and marshal the args through the i64-slot ABI.
/// Returns the compiled unit + its entry args; the caller pushes it to `source` and runs/steps it.
fn dbg_jit_invoke_unit(
    host: &mut Host,
    h: i32,
    code: i32,
    argv: &[i64],
    params: &[ValType],
    results: &[ValType],
) -> Result<(Compiled, Vec<Value>), Trap> {
    let funcs = host.resolve_jit_domain(h).and_then(|domain| {
        let (cd, cu) = host.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
    })?;
    let unit = compile_module(&funcs).ok_or(Trap::Malformed)?;
    let arity_ok = unit
        .sigs
        .first()
        .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
    if !arity_ok {
        return Err(Trap::CapFault);
    }
    let child_args: Vec<Value> = params
        .iter()
        .zip(argv.iter())
        .map(|(ty, s)| slot_to_val(*ty, *s))
        .collect();
    Ok((unit, child_args))
}

/// §22 `Jit.invoke` (op 1) **as a seam-free leaf** — run the invoked unit to completion over the shared
/// `(source, table)` and marshal its returns to `dst…`. Used on the `ScheduledDebugRun` (invoke stays
/// opaque there, as coroutines do) and as production `run_invoke` does. Serviced in [`debug_advance_fiber`].
#[allow(clippy::too_many_arguments)]
fn dbg_jit_invoke_leaf(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    h: i32,
    code: i32,
    argv: &[i64],
    dst: u32,
    params: &[ValType],
    results: &[ValType],
) -> Result<(), Trap> {
    let (unit, child_args) = dbg_jit_invoke_unit(host, h, code, argv, params, results)?;
    let umod = source.push(unit);
    let vals = run_invoke(
        source,
        table,
        umod,
        &child_args,
        fuel,
        mem,
        &mut HostCell::Excl(host),
    )?;
    for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
        let re = slot_to_val(*ty, val_to_slot(*v));
        vt.active.set(dst + i as u32, Reg::from_value(re));
    }
    Ok(())
}

/// §22 `Jit.invoke` (op 1) **as a step-into** (single-vCPU `DebugRun`): compile + push the unit, then
/// arm [`VTask::active_invoke`] so [`debug_advance_fiber`] steps the invoked unit op-by-op (breakpoints
/// fire inside it) instead of running it opaquely — the §22 counterpart of coroutine step-into. The
/// unit runs over the caller's shared window/table; `dst`/`results` marshal its returns back to the
/// caller on completion ([`step_active_invoke`]). `Err` traps the caller (forged handle / bad unit).
#[allow(clippy::too_many_arguments)]
fn dbg_jit_invoke_step_into(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    h: i32,
    code: i32,
    argv: &[i64],
    dst: u32,
    params: &[ValType],
    results: &[ValType],
) -> Result<(), Trap> {
    let (unit, child_args) = dbg_jit_invoke_unit(host, h, code, argv, params, results)?;
    let umod = source.push(unit);
    let cm = source.get(umod).ok_or(Trap::Malformed)?;
    let mut vm = Vm::new(&cm, 0, &child_args)?;
    vm.module = umod;
    let parent_depth = vt.active.stack.len() + 1;
    vt.active_invoke = Some(Box::new(InvokeStep {
        vm,
        dst,
        results: results.into(),
        parent_depth,
    }));
    Ok(())
}

/// Advance the **active §22 invoked unit** (`vt.active_invoke`) by exactly one op — the op-by-op,
/// debugger-facing counterpart of [`run_invoke`]'s loop. The unit runs over the caller's shared
/// `mem`/`host`/`source`/`table` (a seam-free leaf), so its `call_indirect` reaches installed units and
/// any spawn/park/yield/re-invoke is an inert `CapFault` — exactly `run_invoke`'s `_ => CapFault`, only
/// surfaced one op at a time so a breakpoint can fire inside the unit. On the unit's return the caller's
/// `dst…` slots are filled through the i64-slot ABI and control returns to the caller.
fn step_active_invoke(
    vt: &mut VTask,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    enum InvStep {
        Ran,
        Done(Vec<Value>),
        Trap(Trap),
    }
    let step = {
        let iv = vt.active_invoke.as_mut().expect("active invoke present");
        match iv
            .vm
            .resume(source, table, fuel, mem, &mut HostCell::Excl(host), 1)
        {
            Ok(Outcome::Suspended) => InvStep::Ran, // budget boundary — one op done, keep stepping
            Ok(Outcome::Done(vals)) => InvStep::Done(vals),
            // F2 — a punted host call inside an invoked unit keeps the pre-F2 inline wait
            // (`run_invoke`'s arm; the unit is a seam-free atomic leaf, DESIGN §22).
            Ok(Outcome::CapPending { id, dst }) => {
                let r = host.completions().wait(id);
                iv.vm.set(dst, Reg::from_i64(r));
                InvStep::Ran
            }
            // A seam op (spawn/park/yield/cont.*/re-invoke) inside an invoked unit is an inert CapFault,
            // matching run_invoke's `_ => CapFault`.
            Ok(_) => InvStep::Trap(Trap::CapFault),
            Err(t) => InvStep::Trap(t),
        }
    };
    match step {
        InvStep::Ran => FiberStep::Stepped,
        InvStep::Done(vals) => {
            let iv = vt.active_invoke.take().expect("active invoke present");
            for (i, (v, ty)) in vals.iter().zip(iv.results.iter()).enumerate() {
                let re = slot_to_val(*ty, val_to_slot(*v));
                vt.active.set(iv.dst + i as u32, Reg::from_value(re));
            }
            FiberStep::Stepped
        }
        InvStep::Trap(t) => {
            vt.active_invoke = None;
            FiberStep::Trapped(t)
        }
    }
}

/// `memory.wait`: park the caller on futex key `base` until a `notify` or the deadline, unless the
/// value already changed (the compare-under-lock analogue). Mirrors `drive`'s `Wait`.
#[allow(clippy::too_many_arguments)]
fn dbg_wait(
    tasks: &mut [DbgTask],
    ti: usize,
    mem: &Option<Mem>,
    clock: u64,
    base: u64,
    expected: u64,
    width: u32,
    timeout: u64,
    dst: u32,
) {
    let cur = mem
        .as_ref()
        .map(|m| m.atomic_value(base, width))
        .unwrap_or(0);
    if cur != expected {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::WAIT_NOT_EQUAL));
    } else {
        tasks[ti].state = DbgTaskState::BlockedWait {
            key: base,
            deadline: clock.saturating_add(timeout),
            dst,
        };
    }
}

/// `memory.notify`: wake up to `count` waiters on `base` (lowest task index first, deterministic); the
/// woken count lands at `dst`. Mirrors `drive`'s `Notify`.
fn dbg_notify(tasks: &mut [DbgTask], ti: usize, base: u64, count: i32, dst: u32) {
    let want = count as u32;
    let mut woken = 0u32;
    for t in tasks.iter_mut() {
        if woken >= want {
            break;
        }
        if let DbgTaskState::BlockedWait { key, dst: wdst, .. } = t.state {
            if key == base {
                t.vt.active.set(wdst, Reg::from_i32(super::WAIT_WOKEN));
                t.state = DbgTaskState::Runnable;
                woken += 1;
            }
        }
    }
    tasks[ti].vt.active.set(dst, Reg::from_i32(woken as i32));
}

/// SplitMix64 — the stateless mix behind the **seeded pick** (slice 7): the choice at a turn is a
/// pure function of `(seed, turn)`, so any replay — full or from a checkpoint — reproduces the
/// schedule with zero captured scheduler state (INVARIANTS.md #7: recovery never replays captured
/// scheduler records).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The forced-switch override recorded for `turn`, if any (slice 7). Entries are concrete
/// `(turn, task)` pairs resolved at record time, so a replay re-applies the identical choice.
fn forced_at(forced: &[(u64, usize)], turn: u64) -> Option<usize> {
    forced
        .iter()
        .find(|(t, _)| *t == turn)
        .map(|(_, task)| *task)
}

/// The task **pinned** to the scheduler because it is mid-`resume` inside a §14 coroutine body
/// (`active_coro` set). A coroutine `resume` is atomic w.r.t. other vCPUs, so while its body is being
/// stepped op-by-op the scheduler must keep running that same vCPU — never interleaving another thread —
/// until the child yields / faults / returns and `active_coro` clears. At most one task is ever pinned
/// (a task can only enter a coroutine while running, and a pinned task runs alone), so the first match
/// is the pin.
fn dbg_pinned_coro(tasks: &[DbgTask]) -> Option<usize> {
    let _ = tasks;
    None
}

/// Pick the next thread to run under the session's **schedule policy** (slice 7): a forced-switch
/// entry for this `turn` wins (when its task is runnable), then a `seed`ed pick chooses uniformly
/// among the runnable set via [`splitmix64`]`(seed ^ turn)`, else the lowest-index runnable — the
/// original deterministic default. If none is runnable, advance the futex `clock` to the earliest
/// `memory.wait` deadline and wake every timed-out waiter (`WAIT_TIMED_OUT`), then retry. `None`
/// only on a true deadlock (no runnable thread and no waiter) — mirrors `drive`. Every path is a
/// pure function of `(seed, forced, turn, task states)`, so replay reproduces it exactly.
fn dbg_pick_runnable(
    tasks: &mut [DbgTask],
    clock: &mut u64,
    seed: Option<u64>,
    forced: &[(u64, usize)],
    turn: u64,
) -> Option<usize> {
    loop {
        // A forced switch recorded for this turn wins while its task is runnable.
        if let Some(f) = forced_at(forced, turn) {
            if matches!(tasks.get(f).map(|t| &t.state), Some(DbgTaskState::Runnable)) {
                return Some(f);
            }
        }
        let runnable: Vec<usize> = tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.state, DbgTaskState::Runnable))
            .map(|(i, _)| i)
            .collect();
        if !runnable.is_empty() {
            return Some(match seed {
                None => runnable[0], // the original lowest-index default
                Some(s) => runnable[(splitmix64(s ^ turn) % runnable.len() as u64) as usize],
            });
        }
        let next = tasks
            .iter()
            .filter_map(|t| match t.state {
                DbgTaskState::BlockedWait { deadline, .. } => Some(deadline),
                _ => None,
            })
            .min()?;
        *clock = (*clock).max(next);
        for t in tasks.iter_mut() {
            if let DbgTaskState::BlockedWait { deadline, dst, .. } = t.state {
                if deadline <= *clock {
                    t.vt.active.set(dst, Reg::from_i32(super::WAIT_TIMED_OUT));
                    t.state = DbgTaskState::Runnable;
                }
            }
        }
    }
}

impl ScheduledDebugRun {
    /// Open a multithreaded debug session on `m`'s `func(args)`. `None` if the module is outside the
    /// bytecode engine's subset (`compile_module` declines it). The powerbox is empty; use
    /// [`new_with_host`](ScheduledDebugRun::new_with_host) to debug a guest needing a granted capability
    /// (e.g. a §14 `Instantiator` for `instantiate`).
    pub fn new(m: &Module, func: FuncIdx, args: &[Value]) -> Option<ScheduledDebugRun> {
        ScheduledDebugRun::new_with_host(m, func, args, Host::new())
    }

    /// [`new`](ScheduledDebugRun::new) carrying a live powerbox `host`, so a granted `Instantiator`
    /// (`host.grant_instantiator(..)`) reaches the guest as an argument and makes an `instantiate`-using
    /// multithreaded guest debuggable (the debug-scheduler analogue of [`DebugRun::new_with_host`]).
    pub fn new_with_host(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        host: Host,
    ) -> Option<ScheduledDebugRun> {
        m.funcs.get(func as usize)?;
        let ModuleDebug {
            fn_block_base,
            fn_block_types,
            ..
        } = ModuleDebug::build(m, 0);
        let c = compile_module_unfused(&m.funcs)?; // unfused: debug stepping (Slice 5a)
        let dom = Domain::new(c, host.jit_table_log2());
        let mem = build_mem(m);
        let vt = VTask::new(&dom.source.primary(), func as usize, args).ok()?;
        let Domain { source, table } = dom;
        Some(ScheduledDebugRun {
            source,
            table,
            mem,
            host,
            tasks: vec![DbgTask {
                vt,
                threads: Vec::new(),
                env: None,
                state: DbgTaskState::Runnable,
                at_bp: false,
            }],
            extra_envs: Vec::new(),
            fibers: Vec::new(),
            fn_block_base,
            fn_block_types,
            debug: m.debug_info.clone(),
            funcs: std::sync::Arc::from(m.funcs.clone()),
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
            access_sink: None,
            sched_trace: None,
            scheduled_writes: Vec::new(),
            write_cursor: 0,
            sched_seed: None,
            forced: Vec::new(),
            last_watch: None,
            stopped: None,
            focus: 0,
            turn: 0,
            clock: 0,
        })
    }

    /// Replace the run-shared breakpoint set (fires in whichever thread reaches a pc in it).
    pub fn set_breakpoints(&mut self, bps: Vec<super::IrPc>) {
        self.breakpoints = bps;
    }

    /// Replace the run-shared **watchpoints** (DEBUGGING.md W2, cross-thread): each `(addr, len, kind)`
    /// makes the schedule stop *before* any op — in whichever thread — that accesses `[addr, addr+len)`
    /// with a matching read/write kind.
    pub fn set_watchpoints(&mut self, ranges: Vec<(u64, u64, super::WatchKind)>) {
        self.watchpoints = ranges;
    }

    /// Install the run-shared per-op **access sink** ([`AccessSinkFn`]) — fired with the global
    /// `turn` and the executing task index. Observation only; zero cost when never installed.
    pub fn set_access_sink(&mut self, sink: AccessSinkFn) {
        self.access_sink = Some(sink);
    }

    /// Install the session's **scheduled debugger writes** — see `DebugRun::set_scheduled_writes`.
    pub fn set_scheduled_writes(&mut self, mut writes: Vec<(u64, ScheduledWrite)>) {
        writes.sort_by_key(|(c, _)| *c);
        self.write_cursor = writes.partition_point(|(c, _)| *c < self.turn);
        self.scheduled_writes = writes;
    }

    /// The focused task index (the one a `write_var` resolves in) — the backend records it on a
    /// scheduled `Var` write so replays resolve in the same task.
    pub fn focus_task(&self) -> usize {
        self.focus
    }

    /// The shared window's memory-map introspection — see `DebugRun::mem_map_info`.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// Arm (or drop) the **scheduler trace tape** — see [`SchedTraceEvent`]. Arming resets the
    /// tape; observation only, zero cost when off.
    pub fn set_sched_trace(&mut self, on: bool) {
        self.sched_trace = if on { Some(Vec::new()) } else { None };
    }

    /// The trace tape so far (`None` when not armed).
    pub fn sched_trace(&self) -> Option<&[SchedTraceEvent]> {
        self.sched_trace.as_deref()
    }

    /// Set (or clear) the **seeded pick** — see [`ScheduledDebugRun::sched_seed`]. Set before
    /// driving (the DAP backend applies it at construction and on every rebuild).
    pub fn set_sched_seed(&mut self, seed: Option<u64>) {
        self.sched_seed = seed;
    }

    /// Replace the recorded **forced switches** — concrete `(turn, task)` overrides (slice 7).
    pub fn set_forced_switches(&mut self, forced: Vec<(u64, usize)>) {
        self.forced = forced;
    }

    /// The currently-runnable task indices (the forced-switch verb resolves its target from this).
    pub fn runnable_tasks(&self) -> Vec<usize> {
        self.tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.state, DbgTaskState::Runnable))
            .map(|(i, _)| i)
            .collect()
    }

    /// Take the `(addr, write)` of the watchpoint the last stop fired on (cleared by the read), so the
    /// backend can report `StopReason::Watchpoint`. `None` if the last stop was a breakpoint / step.
    pub fn take_watch_hit(&mut self) -> Option<(u64, bool)> {
        self.last_watch.take()
    }

    /// Drive the cooperative schedule until a breakpoint/watchpoint fires (in some thread), the root
    /// finishes, no thread is runnable (`Blocked`), or a thread hits an unsupported op (`Declined`).
    /// Resumable — the previously stopped thread steps one op past its stop before the scan resumes.
    pub fn run_until_stop(&mut self, fuel: &mut u64) -> SchedStop {
        self.drive(fuel, None)
    }

    /// The unified scheduler pump. `step` selects the mode:
    /// - `None` — a plain resume (`continue`/`reverseContinue`): run the **lowest-index** runnable
    ///   thread one op per turn, stopping on any thread's breakpoint or watchpoint.
    /// - `Some((st, max))` — step thread `st`: run **`st`** by preference (falling back to the lowest
    ///   runnable only while `st` is blocked, so a step *over* a `join` can't deadlock), stopping the
    ///   moment `st` reaches a call depth `<= max` at an instruction (`max = None` ⇒ any depth = one
    ///   instruction = step-*in*). Another thread's breakpoint/watchpoint still interrupts a step.
    fn drive(&mut self, fuel: &mut u64, step: Option<(usize, Option<usize>)>) -> SchedStop {
        let Self {
            source,
            table,
            mem,
            host,
            tasks,
            extra_envs,
            fibers,
            funcs,
            breakpoints,
            watchpoints,
            access_sink,
            sched_trace,
            sched_seed,
            forced,
            scheduled_writes,
            write_cursor,
            last_watch,
            fn_block_base,
            fn_block_types,
            debug,
            stopped,
            focus,
            turn,
            clock,
            ..
        } = self;
        *stopped = None;
        loop {
            if let DbgTaskState::Done(res) = &tasks[0].state {
                return SchedStop::Finished(res.clone());
            }
            // A task mid-coroutine is pinned (atomic resume); otherwise prefer the stepping thread while
            // it is runnable (so a step stays on it and a step-over runs its own call), else the
            // lowest-index runnable thread (advancing the futex clock to wake a waiter when the set is
            // stuck; unblocks a stepped `join`/`wait`).
            let pre_pick = sched_trace.as_ref().map(|_| trace_tags(tasks));
            // Precedence: the coroutine pin (an atomicity constraint) > a forced switch recorded
            // for this turn (explicit user intent) > the stepping thread > the policy pick.
            let ti = if let Some(p) = dbg_pinned_coro(tasks) {
                p
            } else if let Some(f) = forced_at(forced, *turn).filter(|f| {
                matches!(
                    tasks.get(*f).map(|t| &t.state),
                    Some(DbgTaskState::Runnable)
                )
            }) {
                f
            } else {
                match step {
                    Some((st, _)) if matches!(tasks[st].state, DbgTaskState::Runnable) => st,
                    _ => match dbg_pick_runnable(tasks, clock, *sched_seed, forced, *turn) {
                        Some(i) => i,
                        None => return SchedStop::Blocked,
                    },
                }
            };
            // Slice 6: the only transition a pick causes is a timed-out wait waking.
            if let (Some(trace), Some(before)) = (sched_trace.as_mut(), pre_pick.as_ref()) {
                trace_pick_diff(before, tasks, *turn, trace);
            }
            // Pre-op stop checks (breakpoint / watchpoint), skipped for a thread that just reported (it
            // must make progress off its current op first, so a loop-body stop re-fires each iteration).
            // Scan the task's *active continuation* — the §14 coroutine child (over its confined window)
            // when this task is mid-`resume`, else its own vCPU — so a breakpoint fires inside a coroutine
            // body on the right thread.
            if !tasks[ti].at_bp {
                let hit = {
                    let cur_vm = tasks[ti].vt.debug_active();
                    let cur_mem: &Option<Mem> = match tasks[ti].env {
                        None => &*mem,
                        Some(k) => &extra_envs[k].mem,
                    };
                    match cur_vm.cur_ir_pc(source) {
                        Some(pc) if breakpoints.contains(&pc) => Some((pc, None)),
                        Some(pc) if !watchpoints.is_empty() && pc.module == 0 => watch_hit_before(
                            cur_vm,
                            cur_mem,
                            funcs,
                            fn_block_base,
                            watchpoints,
                            pc.func,
                            pc.block,
                            pc.inst,
                        )
                        .map(|w| (pc, Some(w))),
                        _ => None,
                    }
                };
                if let Some((pc, watch)) = hit {
                    let reason = match watch {
                        Some((addr, write)) => {
                            *last_watch = Some((addr, write));
                            SchedBreak::Watchpoint { addr, write }
                        }
                        None => SchedBreak::Breakpoint,
                    };
                    tasks[ti].at_bp = true;
                    *stopped = Some(ti);
                    *focus = ti;
                    return SchedStop::Break { pc, reason };
                }
            }
            apply_due_writes_sched(
                scheduled_writes,
                write_cursor,
                *turn,
                tasks,
                source,
                mem,
                debug.as_ref(),
                fn_block_base,
                fn_block_types,
            );
            if let Some(sink) = access_sink.as_mut() {
                let cur_vm = tasks[ti].vt.debug_active();
                emit_access(cur_vm, source, funcs, fn_block_base, *turn, ti, sink);
            }
            // Slice 6: the turn record + the pre-advance snapshot the park/wake differ compares.
            let trace_turn = *turn;
            let pre_adv = sched_trace.as_ref().map(|_| trace_tags(tasks));
            if let Some(trace) = sched_trace.as_mut() {
                trace.push(SchedTraceEvent::Turn {
                    turn: trace_turn,
                    task: ti,
                });
            }
            if let Serviced::Declined = service_advance(
                tasks, ti, extra_envs, fibers, source, table, fuel, mem, host, *clock, turn,
            ) {
                // A coroutine / tier-up op this engine does not drive — bail, `turn` untouched.
                return SchedStop::Declined;
            }
            // Slice 6: derive the park/wake/spawn edges this advance caused (see `trace_diff`).
            if let (Some(trace), Some(before)) = (sched_trace.as_mut(), pre_adv.as_ref()) {
                trace_diff(before, tasks, trace_turn, ti, trace);
            }
            // Post-op step target: the stepping thread reached a qualifying call depth at an instruction.
            // Depth is cumulative across a coroutine boundary (the child's frames sit above the parent's
            // resume frame), so a step-over of a `resume` runs the child to completion and a step inside a
            // coroutine body compares child-local frames — mirroring the single-vCPU `DebugRun::step_to`.
            if let Some((st, max_depth)) = step {
                if ti == st && matches!(tasks[st].state, DbgTaskState::Runnable) {
                    let cur_vm = tasks[st].vt.debug_active();
                    let depth = cur_vm.stack.len() + 1;
                    if max_depth.is_none_or(|m| depth <= m) {
                        if let Some(pc) = cur_vm.cur_ir_pc(source) {
                            *stopped = Some(st);
                            *focus = st;
                            return SchedStop::Break {
                                pc,
                                reason: SchedBreak::Step,
                            };
                        }
                    }
                }
            }
        }
    }

    /// Step the stopped thread until its call depth is `<= max_depth` (`None` ⇒ any = one instruction),
    /// keeping other threads frozen unless the stepped thread blocks. The shared driver for the stepping
    /// verbs — mirrors `DebugRun::step_to`.
    fn step_to(&mut self, max_depth: Option<usize>, fuel: &mut u64) -> SchedStop {
        let Some(st) = self.stopped else {
            return self.run_until_stop(fuel);
        };
        self.tasks[st].at_bp = true; // step *off* the current op first, then seek the next stop
        self.drive(fuel, Some((st, max_depth)))
    }

    /// **Step** one instruction — descends into a call — the multithreaded counterpart of
    /// `DebugRun::step`. Drives the stopped thread; other threads stay frozen.
    pub fn step(&mut self, fuel: &mut u64) -> SchedStop {
        self.step_to(None, fuel)
    }

    /// The stopped thread's **cumulative** call depth (the child's frames count above the parent's resume
    /// frame while it is mid-coroutine — see [`DebugRun::step_depth`]). Used by the depth-bounded verbs so
    /// they treat a coroutine `resume` boundary like an ordinary call.
    fn step_depth(&self, s: usize) -> usize {
        let cur_vm = self.tasks[s].vt.debug_active();
        cur_vm.stack.len() + 1
    }

    /// **Step over** the next source op: run any call it makes to completion (schedule advances only if
    /// the stepped thread blocks), landing at the next op at the same call depth.
    pub fn step_over(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self.stopped.map(|s| self.step_depth(s));
        self.step_to(max, fuel)
    }

    /// **Step out** — run until the stepped thread's current function returns (one call depth shallower).
    pub fn step_out(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self.stopped.map(|s| self.step_depth(s).saturating_sub(1));
        self.step_to(max, fuel)
    }

    /// Advance the schedule by exactly one visible op (the raw time quantum for replay-based reverse
    /// `seek` — DEBUGGING.md W1), honoring **no** breakpoint/watch/step checks: the lowest-index runnable
    /// thread runs one op, `turn` ticks. Returns `false` once the root has finished (or the schedule can
    /// no longer advance — blocked/unsupported). Because the debug schedule is deterministic (pure
    /// compute, one-op-per-turn, lowest-index pick), replaying `t` ticks from a fresh session reproduces
    /// the exact state at global turn `t`.
    pub fn tick(&mut self, fuel: &mut u64) -> bool {
        if matches!(self.tasks[0].state, DbgTaskState::Done(_)) {
            return false;
        }
        let Self {
            source,
            table,
            mem,
            host,
            tasks,
            extra_envs,
            fibers,
            turn,
            clock,
            funcs,
            fn_block_base,
            fn_block_types,
            debug,
            access_sink,
            sched_trace,
            sched_seed,
            forced,
            scheduled_writes,
            write_cursor,
            ..
        } = self;
        // A task mid-coroutine is pinned (atomic resume — the same vCPU runs the whole body); the same
        // pin on replay reconstructs the coroutine's op sequence deterministically. The policy pick
        // (seed + forced) matches `drive`'s, so a tick-replay reproduces the interactive schedule.
        let pre_pick = sched_trace.as_ref().map(|_| trace_tags(tasks));
        let Some(ti) = dbg_pinned_coro(tasks)
            .or_else(|| dbg_pick_runnable(tasks, clock, *sched_seed, forced, *turn))
        else {
            return false; // no runnable thread and no waiter (deadlock) — can't advance
        };
        if let (Some(trace), Some(before)) = (sched_trace.as_mut(), pre_pick.as_ref()) {
            trace_pick_diff(before, tasks, *turn, trace);
        }
        apply_due_writes_sched(
            scheduled_writes,
            write_cursor,
            *turn,
            tasks,
            source,
            mem,
            debug.as_ref(),
            fn_block_base,
            fn_block_types,
        );
        if let Some(sink) = access_sink.as_mut() {
            let cur_vm = tasks[ti].vt.debug_active();
            emit_access(cur_vm, source, funcs, fn_block_base, *turn, ti, sink);
        }
        let trace_turn = *turn;
        let pre_adv = sched_trace.as_ref().map(|_| trace_tags(tasks));
        if let Some(trace) = sched_trace.as_mut() {
            trace.push(SchedTraceEvent::Turn {
                turn: trace_turn,
                task: ti,
            });
        }
        if let Serviced::Declined = service_advance(
            tasks, ti, extra_envs, fibers, source, table, fuel, mem, host, *clock, turn,
        ) {
            // An unsupported op — tick this engine's clock (as it did unconditionally before) and
            // stop the replay here.
            *turn += 1;
            return false;
        }
        // Slice 6: the park/wake/spawn edges this replayed op caused (identical to `drive`'s,
        // so a `tick`-replay refills the tape deterministically).
        if let (Some(trace), Some(before)) = (sched_trace.as_mut(), pre_adv.as_ref()) {
            trace_diff(before, tasks, trace_turn, ti, trace);
        }
        !matches!(tasks[0].state, DbgTaskState::Done(_))
    }

    /// The current global turn (visible ops replayed so far) — the reverse-`seek` coordinate.
    pub fn op_turn(&self) -> u64 {
        self.turn
    }

    /// The powerbox host backing this run — for reading effects a debugged multithreaded guest
    /// produced (captured stdout) and its [`CapTape`](Host::cap_tape) so a reverse `seek` rebuild
    /// replays identical cap inputs. The scheduled-engine twin of [`DebugRun::host`].
    pub fn host(&self) -> &Host {
        &self.host
    }
    /// Mutable powerbox host — e.g. to drain captured stdout between stops.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Position the session at the current schedule point after a raw `tick`-replay `seek`: the stopped +
    /// focused thread becomes the one about to run (lowest-index runnable), or none once the run finished.
    pub fn locate(&mut self) {
        let next = self
            .tasks
            .iter()
            .position(|t| matches!(t.state, DbgTaskState::Runnable));
        self.stopped = next;
        self.focus = next.unwrap_or(0);
    }

    /// After a `seek` landed exactly on a breakpoint op, arm the stopped thread's skip so a forward
    /// resume steps past it instead of immediately re-reporting the same stop.
    pub fn arm_breakpoint_skip(&mut self) {
        if let Some(st) = self.stopped {
            self.tasks[st].at_bp = true;
        }
    }

    /// Whether the scheduled continuation is fully captured by the per-task active `Vm`s + the shared
    /// window bytes + the host substate + the scheduler clocks — the subset a multi-vCPU time-travel
    /// **checkpoint** (W1) snapshots. Mirrors [`DebugRun::checkpointable`], extended over every task:
    /// **§12 fibers** are admitted (the run-shared registry + each task's active fiber / resume chain,
    /// all sharing the run's window) except an event-parked (`memory.wait`) fiber (non-deterministic
    /// wall-clock deadline); **§14 coroutines** are admitted (not demand, pristine `nested_view`),
    /// same-module *or* separate-module (a separate module's pushed unit rides in `extra_units`); and
    /// **§14 `instantiate` / `instantiate_module` children** are admitted (each [`DbgEnv`] a `nested_view`
    /// over the shared backing + a deterministic `Instantiator`/`AddressSpace` powerbox + a natural table
    /// over the child's module, rebuilt on restore) — this is what admits scheduled coroutines, which only
    /// ever arise alongside an `instantiate` sibling (the bytecode engine rejects `coroutine + thread`).
    /// Both coroutines and children may be **demand**/self-page-mapping: each child's own page map is
    /// captured (`child_checkpointable`: `layout_snapshot_safe`, no §13 regions, within the parent's
    /// prefix) and its bytes ride in the shared snapshot. Still excluded (→ replay-from-turn-0): a
    /// region-aliased child, or one carved beyond the parent's captured prefix.
    fn checkpointable(&self) -> bool {
        self.host.checkpoint_safe()
            && self.mem.as_ref().is_none_or(|m| m.layout_snapshot_safe())
            && !self.fibers.iter().any(|f| {
                matches!(
                    f,
                    FiberState::WaitParked { .. } | FiberState::CapParked { .. }
                )
            })
            && self.extra_envs.iter().all(|e| {
                e.host.checkpoint_safe() && child_checkpointable(e.mem.as_ref(), self.mem.as_ref())
            })
    }

    /// Snapshot the scheduled continuation at the current [`turn`](ScheduledDebugRun::op_turn) for the
    /// backend's checkpoint ladder — `None` outside the [`checkpointable`](ScheduledDebugRun::checkpointable)
    /// subset. Captures each task's active `Vm` + join table + state, the shared window bytes, the host
    /// replay substate, and both scheduler clocks; the transient `stopped`/`focus`/`last_watch` are
    /// *not* captured — [`locate`](ScheduledDebugRun::locate) rederives them from the task states.
    pub fn snapshot(&self) -> Option<ScheduledSnapshot> {
        if !self.checkpointable() {
            return None;
        }
        Some(ScheduledSnapshot {
            turn: self.turn,
            clock: self.clock,
            tasks: self
                .tasks
                .iter()
                .map(|t| DbgTaskSnapshot {
                    active: t.vt.active.clone(),
                    active_id: t.vt.active_id,
                    chain: t.vt.chain.clone(),
                    root_shadow_sp: t.vt.root_shadow_sp,
                    threads: t.threads.clone(),
                    env: t.env,
                    state: t.state.clone(),
                    at_bp: t.at_bp,
                })
                .collect(),
            fibers: self.fibers.clone(),
            // Each child env's module = the module its owning task runs (`0` = same-module `instantiate`;
            // `>= 1` = a separate-module `instantiate_module` child), so its table rebuilds correctly.
            extra_envs: self
                .extra_envs
                .iter()
                .enumerate()
                .map(|(k, e)| {
                    let module = self
                        .tasks
                        .iter()
                        .find(|t| t.env == Some(k))
                        .map_or(0, |t| t.vt.active.module);
                    env_snapshot(e, module)
                })
                .collect(),
            extra_units: self.source.extra_units(),
            mem: self.mem.as_ref().map(|m| m.layout_snapshot()),
            host: self.host.replay_substate(),
        })
    }

    /// Restore a [`snapshot`](ScheduledDebugRun::snapshot) into this **freshly built** run (its
    /// breakpoints/watchpoints re-armed by the backend before this call), so a subsequent `tick`-replay
    /// resumes exactly at the snapshot's global turn rather than turn 0. Rebuilds the task set (each a
    /// root-only `VTask` around the captured active `Vm`), reseeds the shared window bytes, restores the
    /// host substate and both scheduler clocks, and clears the transient stop state (`locate` rederives
    /// it). A separate-module coroutine/child's pushed source units are re-pushed first (so its `module`
    /// index resolves); the run-shared fibers and each child env are rebuilt from the snapshot.
    pub fn restore(&mut self, snap: &ScheduledSnapshot) {
        self.fibers = snap.fibers.clone();
        // Re-push any separate-module units before rebuilding envs/coroutines (their `module` indices
        // resolve against the source).
        self.source.reset_extra(&snap.extra_units);
        if let (Some(m), Some(layout)) = (self.mem.as_mut(), snap.mem.as_ref()) {
            m.restore_layout(layout);
        }
        // Rebuild each task's full `VTask` and each §14 `instantiate`-child env. Coroutine and child
        // windows are `nested_view`s over the just-reseeded shared window (their bytes — shared via the
        // backing region — are already correct); each table is rebuilt over the child's own module.
        let shared_mem = self.mem.as_ref();
        let source = &*self.source;
        self.extra_envs = snap
            .extra_envs
            .iter()
            .map(|es| rebuild_env(es, shared_mem, source))
            .collect();
        self.tasks = snap
            .tasks
            .iter()
            .map(|ts| {
                DbgTask {
                    vt: VTask {
                        active: ts.active.clone(),
                        active_id: ts.active_id,
                        chain: ts.chain.clone(),
                        root_shadow_sp: ts.root_shadow_sp,
                        active_invoke: None, // scheduled engine keeps invoke a leaf (never steps in)
                        invoke_step_into: false,
                    },
                    threads: ts.threads.clone(),
                    env: ts.env,
                    state: ts.state.clone(),
                    at_bp: ts.at_bp,
                }
            })
            .collect();
        self.host.restore_replay_substate(&snap.host);
        self.turn = snap.turn;
        self.clock = snap.clock;
        self.stopped = None;
        self.focus = 0;
        self.last_watch = None;
    }

    /// The run's result once the root has finished (`None` while still running).
    pub fn result(&self) -> Option<&Result<Vec<Value>, Trap>> {
        match &self.tasks[0].state {
            DbgTaskState::Done(r) => Some(r),
            _ => None,
        }
    }

    /// Every live (not-yet-finished) vCPU — one DAP thread each. The stopped thread is among them.
    pub fn threads(&self) -> Vec<u64> {
        (0..self.tasks.len())
            .filter(|&i| !matches!(self.tasks[i].state, DbgTaskState::Done(_)))
            .map(|i| i as u64)
            .collect()
    }

    /// The thread index currently paused on a breakpoint (drives stepping); `None` while running.
    pub fn stopped_task(&self) -> Option<u64> {
        self.stopped.map(|i| i as u64)
    }

    /// Focus read-inspection (`backtrace`/`read_var`/`read_window`) on a live thread; `false` if `id`
    /// is not a live task. Resets to the stopped thread on the next `run_until_stop`.
    pub fn select_task(&mut self, id: u64) -> bool {
        let i = id as usize;
        if i < self.tasks.len() && !matches!(self.tasks[i].state, DbgTaskState::Done(_)) {
            self.focus = i;
            true
        } else {
            false
        }
    }

    /// The scheduled-mode logical clock (visible ops across all vCPUs).
    pub fn turn(&self) -> u64 {
        self.turn
    }

    /// The memory window a task steps against: its confined `instantiate` env (`Some(k)`) or the shared
    /// mem — so inspection of a focused child reads its own confined window.
    fn task_mem(&self, ti: usize) -> &Option<Mem> {
        match self.tasks[ti].env {
            None => &self.mem,
            Some(k) => &self.extra_envs[k].mem,
        }
    }

    /// A [`FrameReader`] over the **focused** thread's currently-stepping `Vm` (what `select_task` chose):
    /// the thread's own vCPU over its window (a confined `instantiate` child reads its `nested_view`).
    fn reader(&self) -> FrameReader<'_> {
        let vm = self.tasks[self.focus].vt.debug_active();
        FrameReader {
            vm,
            source: &self.source,
            mem: self.task_mem(self.focus),
            debug: self.debug.as_ref(),
            fn_block_base: &self.fn_block_base,
            fn_block_types: &self.fn_block_types,
            // A separate-module coroutine on the scheduled engine carries its own §6 metadata (built at
            // spawn); a same-module one leaves it `None` (its frames are module 0, read against the fields
            // above).
            coro_debug: None,
        }
    }

    /// Call-stack depth of the focused thread.
    pub fn depth(&self) -> usize {
        self.reader().depth()
    }

    /// The `IrPc` of the focused thread's frame `depth` levels from the top.
    pub fn frame_pc(&self, depth: usize) -> Option<super::IrPc> {
        self.reader().frame_pc(depth)
    }

    /// Read a source variable by name in the focused thread's frame `depth` levels from the top.
    pub fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        self.reader().read_var(depth, name, width)
    }

    /// The window address of a memory-located source variable in the focused thread's frame `depth`.
    pub fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        self.reader().var_addr(depth, name)
    }

    /// Write a source variable in the **focused** thread's frame — see `DebugRun::write_var`.
    pub fn write_var(&mut self, depth: usize, name: &str, value: i64, width: usize) -> bool {
        let focus = self.focus;
        if self.tasks.get(focus).is_none() {
            return false;
        }
        let Some(target) = self.reader().write_target(depth, name) else {
            return false;
        };
        match target {
            WriteTarget::Ssa { reg, ty } => {
                let v = match ty {
                    ValType::I32 => Value::I32(value as i32),
                    ValType::I64 => Value::I64(value),
                    _ => return false,
                };
                match self.tasks[focus].vt.active.regs.get_mut(reg) {
                    Some(r) => {
                        *r = Reg::from_value(v);
                        true
                    }
                    None => false,
                }
            }
            WriteTarget::Win { addr } => {
                let w = width.clamp(1, 8);
                self.write_window(addr, &value.to_le_bytes()[..w])
            }
        }
    }

    /// Write bytes into the shared guest window — see `DebugRun::write_window`.
    pub fn write_window(&mut self, addr: u64, bytes: &[u8]) -> bool {
        self.mem
            .as_mut()
            .and_then(|m| m.write_bytes(addr, bytes))
            .is_some()
    }

    /// Read `len` bytes from the focused thread's guest window at `addr`: the active coroutine child's
    /// confined window when mid-`resume`, else the thread's own window (its confined `instantiate`
    /// window or the shared mem).
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match self.task_mem(self.focus).as_ref() {
            Some(m) => m.read_window(addr, len),
            None => Err(Trap::Malformed),
        }
    }
}

/// Like [`compile_and_run`], but drives the reified [`Vm`] in slices of at most `slice` ops,
/// suspending and resuming at op boundaries until the entry function completes (or traps). The
/// result must be **bit-identical** to [`compile_and_run`] for any `slice ≥ 1` — that equality is
/// what proves the suspend/resume machinery (Slice 1c-2) preserves the continuation exactly. Test
/// surface for the "interrupt-anywhere" harness; not a production entry point.
pub fn compile_and_run_sliced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    slice: u64,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m);
    let mut host = Host::new();
    Some(drive(
        dom,
        func,
        args,
        fuel,
        &mut mem,
        &mut host,
        slice.max(1),
    ))
}

fn run(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    // The production path never preempts itself: an unlimited budget makes `resume` run straight to
    // completion, with the per-op budget branch perfectly predicted (so the hot loop is unchanged).
    drive(dom, entry, args, fuel, mem, host, u64::MAX)
}

/// Why [`Vm::resume`] returned. `Done`/`Suspended` are the run-to-completion + budget cases; the
/// `Cont*`/`Suspend` cases are §12 fiber switches handled within [`step_vcpu`] (a vCPU's own fiber
/// registry); the `Thread*`/`Memory*` cases are §12 multi-vCPU events handled by the [`drive`]
/// scheduler. A trap is the `Err` arm of `resume`'s `Result` and is terminal, like the tree-walker.
enum Outcome {
    Done(Vec<Value>),
    Suspended,
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): a direct `Call` to an eligible module-0
    /// function. The host runs the emitted `f{func}` region and delivers its `n_results` results to
    /// the absolute register slot `dst`. `argv` is the marshalled arguments (raw i64 slots).
    /// `mapped` is the window's scalar committed extent at call entry ([`Mem::scalar_extent`]) — the
    /// host MUST write it to the emitted module's `"mapped"` global before invoking `f{func}`, so the
    /// emitted bounds check admits exactly what the interpreter would (#717). Surfaced only when the
    /// extent is scalar-representable; otherwise the call is interpreted (fail-closed decline).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
        mapped: u64,
    },
    /// F2 (FIBER_PARK.md) — a punted offloadable dispatch (`Pending(completion_id)`) with an
    /// exactly-`i64` reply, surfaced so the DRIVER decides the wait shape: the cooperative
    /// `drive` parks a punting FIBER (`FiberState::CapParked` — the slice-5a contract) and
    /// blocks inline at root; every other driver keeps the slice-1 inline wait (the I45
    /// posture). The op already advanced `pc`; delivery writes the scalar to `dst`.
    CapPending {
        id: u64,
        dst: u32,
    },
    /// `cont.new`: register a fiber for `(funcref, sp)`, write its handle to `dst`, continue.
    ContNew {
        funcref: i32,
        sp: i64,
        dst: u32,
    },
    /// `cont.resume`: switch into fiber `kh` with `arg`; `(status, value)` land at `dst`/`dst+1`.
    /// `blocking` marks the I48 `cont.resume.block` variant; `resume_ip` is this op's own program
    /// counter, so a blocking park can rewind the resumer's cursor to re-execute the resume on wake.
    ContResume {
        kh: i32,
        arg: i64,
        dst: u32,
        blocking: bool,
        resume_ip: usize,
    },
    /// `suspend`: hand `value` to the resumer; the parked fiber's `dst` receives the next resume arg.
    FiberSuspend {
        value: i64,
        dst: u32,
    },
    /// `thread.spawn`: spawn a vCPU running `func(sp, arg)`; its handle lands at `dst`. `module` is
    /// the **spawning frame's** module — `func` resolves there, so an installed §22 unit's code
    /// spawns the *unit's own* functions (CONSOLIDATION.md §11), and the child's root frame starts
    /// in that module.
    ThreadSpawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
        module: usize,
    },
    /// `thread.join`: park until child `handle` finishes; its result (or trap) lands at `dst`.
    ThreadJoin {
        handle: i32,
        dst: u32,
    },
    /// §3.6 (I36 slice 2) — a caller's `cap.call` through a live-callee offer. The dispatch is
    /// already enqueued on the callee (the op exec holds the callee `Arc`); the driver parks this
    /// task on `ticket` until the callee's serve loop settles the completion cell, then delivers
    /// the reply to `dst`. The cursor is persisted PAST the op (the reply is the call's result).
    LiveCall {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// §3.6 (I36 slice 2) — `svc.wait` with an empty queue and no progress: park this task on its
    /// domain until a caller's enqueue re-admits it. The cursor is persisted AT the op, so the
    /// wake re-executes the whole serve drain (the tree-walker's rewound park).
    SvcWait,
    /// §3.6 (I36 slice 2) — `child_offer`: mint a live offer over child `child`'s export
    /// `export` (driver-side — it owns the child envs); the handle (or `-EINVAL`) lands at `dst`.
    ChildOffer {
        child: i32,
        export: u32,
        dst: u32,
    },
    /// FORK.md §9.2 — `clone_caller`: fork the caller parked on the running handler's dispatch into a
    /// twin. Driver-side (it owns the task/env set + the parked caller). `reply_orig` = `Some` in the
    /// explicit two-reply form, `None` in pid mode (the parent gets the twin's task id). The driver
    /// reads the handler's `serve_ticket` to name the caller; the twin handle (or an errno) lands at
    /// `dst` when `has_result`.
    CloneCaller {
        reply_orig: Option<i64>,
        reply_twin: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap`: reap twin `pid` on behalf of the caller parked on this handler's
    /// dispatch ([`Outcome::Reap`]).
    Reap {
        pid: i64,
        dst: u32,
        has_result: bool,
    },
    /// §14 `Instantiator.instantiate`: the authority `(ibase, isize)` is resolved; the driver builds a
    /// **confined executor child** running entry `entry` over `[ibase+off, +2^size_log2)` with its own
    /// attenuated powerbox and `quota` fuel, registers it (handle = thread slot), and writes the handle
    /// (or `EINVAL`) to `dst`. Unlike a coroutine, the child runs on the scheduler — joinable via the
    /// shared thread machinery (`Instantiator.join` compiles to [`Outcome::ThreadJoin`]).
    Instantiate {
        ibase: u64,
        isize: u64,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        dst: u32,
        /// op 11 (`instantiate_named`): the grant-list `(ptr, count)` (op 0 is `None`), read from the
        /// register operands so the driver can re-grant the named caps into the child's powerbox.
        grants: Option<(u64, u64)>,
        /// §3d (op 17): the record's `Budget` handle (`0` = none). The driver **funds** the child
        /// from it at its commit site — peek-then-drain, so a refused spawn leaves it intact.
        budget: i32,
    },
    /// §14 `Instantiator.instantiate_module`: like [`Outcome::Instantiate`], plus the resolved
    /// `Module` handle `mh` whose granted program the child runs (the driver resolves + compiles it).
    InstantiateModule {
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        dst: u32,
        /// op 13 `instantiate_module_named`: the resolved `(grants_ptr, grants_n)` window coordinates
        /// of the child's by-name grant list (op 5 is `None`).
        grants: Option<(u64, u64)>,
        /// §3d (op 17): the record's `Budget` handle (`0` = none) — see [`Outcome::Instantiate`].
        budget: i32,
    },
    /// `memory.wait`: futex wait on confined address `base` (already validated); `dst` gets the
    /// status (0 woken / 1 not-equal / 2 timed-out).
    MemoryWait {
        base: u64,
        expected: u64,
        width: u32,
        timeout: u64,
        dst: u32,
    },
    /// `memory.notify`: wake up to `count` waiters on `base`; the woken count lands at `dst`.
    MemoryNotify {
        base: u64,
        count: i32,
        dst: u32,
    },
    /// §22 `install`: the `Jit` cap `h` is authority for code-handle `code`; the driver compiles +
    /// installs the unit and writes the slot (or `-ENOSPC`) to `dst`.
    JitInstall {
        h: i32,
        code: i32,
        dst: u32,
    },
    /// §22 `uninstall`: clear table `slot` (authority `h`); `0`/`EINVAL` → `dst`.
    JitUninstall {
        h: i32,
        slot: i64,
        dst: u32,
    },
    /// §22 `invoke`: run code-handle `code` over the shared window; `argv` are the args as i64 slots,
    /// `params`/`results` type them for the slot ABI; results → `dst…`.
    JitInvoke {
        h: i32,
        code: i32,
        argv: Box<[i64]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §GC `gc.roots`: operands already resolved + the `mask` validated. The driver does the scan
    /// (it owns the resume chain / fiber registry / coroutines), writes the buffer, and delivers the
    /// total to `dst`.
    GcRoots {
        lo: u64,
        hi: u64,
        mask: u64,
        buf: u64,
        cap: usize,
        dst: u32,
    },
    /// **Blocking stdin park**: a `Stream{In}` `read` found the buffer exhausted under
    /// [`Host::set_stdin_blocking`]. The read did not complete and `pc` was *not* advanced, so the
    /// driver re-issues it after more input arrives. Only the resumable [`Vcpu`] driver honours this
    /// (surfacing [`VcpuEvent::StdinPark`]); the one-shot / scheduler drivers never opt into blocking
    /// stdin, so it never reaches them.
    StdinPark,
}

/// Monotonic clock for a fiber's **real** (busy-poll) `memory.wait` timeout. Returns nanoseconds
/// from an arbitrary process epoch.
///
/// - **Native**: real monotonic wall time (a process-global base [`Instant`]). A timed wait polled
///   in a busy `cont.resume` loop fires after the requested duration elapses, as before.
/// - **Wasm** (`wasm32-unknown-unknown`): there is **no wall clock** — `Instant::now()` panics
///   (`std::time::Instant::now` → `unreachable`). The cdylib is deliberately import-free, so we
///   have no host time either. Instead this returns a monotonic **poll counter** that advances one
///   tick per observation, and [`sched_wall_deadline`] arms the timeout at a small fixed number of
///   ticks ([`WASM_WAIT_POLL_TICKS`]). The busy resume-poll loop therefore still terminates (the
///   sole job of the real deadline; the deterministic *logical* `deadline` remains the idle-time
///   timer). Wall-clock fidelity is meaningless on this target, so counting polls is the honest
///   substitute and keeps the confinement/verifier paths untouched.
#[cfg(not(target_family = "wasm"))]
fn sched_wall_now() -> u64 {
    use std::time::Instant;
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
}
#[cfg(target_family = "wasm")]
fn sched_wall_now() -> u64 {
    use core::sync::atomic::{AtomicU64, Ordering};
    static TICKS: AtomicU64 = AtomicU64::new(0);
    TICKS.fetch_add(1, Ordering::Relaxed)
}

/// Poll-ticks a wasm timed `memory.wait` waits before its busy-poll timeout fires (see
/// [`sched_wall_now`]). Small so a `sleep` resolves promptly; non-zero so sibling fibers still get
/// a few turns first rather than the sleeper resolving on its very first poll.
#[cfg(target_family = "wasm")]
const WASM_WAIT_POLL_TICKS: u64 = 8;

/// Arm a real (busy-poll) timeout `timeout` nanoseconds out — native uses the real duration; wasm
/// uses a small fixed tick budget (wall time is meaningless there). See [`sched_wall_now`].
#[cfg(not(target_family = "wasm"))]
fn sched_wall_deadline(timeout: u64) -> u64 {
    sched_wall_now().saturating_add(timeout)
}
#[cfg(target_family = "wasm")]
fn sched_wall_deadline(_timeout: u64) -> u64 {
    sched_wall_now().saturating_add(WASM_WAIT_POLL_TICKS)
}

/// A §12 fiber's state in the driver's per-vCPU registry (handle = index). A durable run maintains the
/// per-context shadow-SP swap ([`shadow_switch`]) and, on freeze, flattens each `Parked` fiber into its
/// shadow region ([`freeze_drive`]); on thaw a flattened fiber is re-seeded as `Pending`. `Clone` for
/// time-travel checkpointing (W1): a fiber-carrying `DebugRun` snapshots its whole registry — each
/// fiber `Vm` shares the one window (snapshotted separately), so a clone is a faithful deep copy.
#[derive(Clone)]
enum FiberState {
    /// Created by `cont.new` but never resumed: starts by calling `funcref(sp, arg)`.
    Pending { funcref: i32, sp: i64 },
    /// Suspended mid-run; resuming delivers the new `arg` into `suspend_dst` and continues `vm`.
    Parked { vm: Vm, suspend_dst: u32 },
    /// §3.6 slice 5a — **event-parked on a futex wait**: the fiber's `memory.wait` parked the
    /// FIBER, not its vCPU (the tree-walk oracle's fiber-park routing, `fiber_parks.rs`). Not
    /// resumable until an event sets `woken`; a `cont.resume` meanwhile reports `FIBER_PARKED`
    /// to the resumer without switching (the cooperative poll). Woken by `notify`
    /// (`WAIT_WOKEN`), by the park-time value recheck (`WAIT_NOT_EQUAL` — after one transient
    /// `FIBER_PARKED`, matching the oracle's register-then-recheck), or by its timeout — which
    /// fires at driver idle via the **logical** `deadline` (with the whole-vCPU wait timers) or
    /// at a `cont.resume` poll via the **real** `real_deadline`, so a busy resume-poll loop
    /// terminates without depending on driver idle time (the jacl timed-wait shape; see
    /// `svm/tests/fiber_timed_wait.rs`).
    WaitParked {
        vm: Vm,
        /// The wait's status register in `vm`; the waking resume writes the `WAIT_*` result here.
        wait_dst: u32,
        /// The confined wait address (the same key `TaskState::BlockedWait` parks on).
        key: u64,
        /// Logical-clock deadline (`clock + timeout`), fired when no task is runnable.
        deadline: u64,
        /// Real-clock deadline (nanoseconds from [`sched_wall_now`]'s epoch), checked at each
        /// `cont.resume` poll of this fiber. Native: monotonic wall time. Wasm: a monotonic
        /// poll counter (no wall clock on `wasm32-unknown-unknown`), so a busy resume-poll loop
        /// still terminates — see [`sched_wall_now`].
        real_deadline: u64,
        /// `Some(status)` once the event fired — the fiber is claimable and the next resume
        /// delivers the status; `None` while still blocked.
        woken: Option<i32>,
    },
    /// F2 (FIBER_PARK.md) — **event-parked on a punt completion**: the fiber's blocking host
    /// call punted to the offload pool (`Pending(completion_id)`) and parked the FIBER, not its
    /// vCPU (the oracle's `CapPending` fiber park, `fiber_parks.rs`). Not resumable until the
    /// completion is claimed into `woken`; a `cont.resume` meanwhile reports `FIBER_PARKED`
    /// without switching (the cooperative poll). Claims happen ONLY through the ordered drain
    /// (`drain_cap_parked` — smallest outstanding id first, stop at the first not-yet-arrived),
    /// so a later completion never overtakes an earlier parked fiber: bit-exact with the
    /// oracle's `completion_drain` (the §18 pin). The drain runs at each `cont.resume` poll of
    /// a cap-parked fiber and at driver idle (which blocks on the completion store when only
    /// cap parks remain — a pool completion is pending work, never a deadlock).
    CapParked {
        vm: Vm,
        /// The `cap.call`'s result register in `vm`; the waking resume writes the scalar here.
        dst: u32,
        /// The completion id this fiber waits on (ids are minted monotonically — submission
        /// order — so "smallest outstanding" is the delivery order).
        id: u64,
        /// `Some(result)` once the drain claimed the completion; `None` while still in flight.
        woken: Option<i64>,
    },
    /// Currently on the resume chain (active or an ancestor) — not independently resumable.
    /// `blocking_ip` (I48): `Some(ip)` if this fiber's current resume used `cont.resume.block`, so a
    /// park inside it idles the resumer (rewinding the resumer's cursor to `ip`) instead of returning
    /// `FIBER_PARKED`; `None` for a plain `cont.resume`. Set at the claim, read when the fiber parks.
    Running { blocking_ip: Option<usize> },
    /// Returned; resuming again is a `FiberFault`.
    Done,
}

/// F2 (FIBER_PARK.md) — the ordered completion drain over the fiber registry: claim ready punt
/// completions for [`FiberState::CapParked`] fibers **smallest-id-first, stopping at the first
/// not-yet-arrived** — the bytecode mirror of the oracle scheduler's `completion_drain`
/// (submission-ordered delivery, the §18 pin). Returns whether anything was claimed.
fn drain_cap_parked(fibers: &mut [FiberState], comps: &super::Completions) -> bool {
    let mut claimed = false;
    loop {
        let Some((fi, id)) = fibers
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f {
                FiberState::CapParked {
                    id, woken: None, ..
                } => Some((i, *id)),
                _ => None,
            })
            .min_by_key(|&(_, id)| id)
        else {
            return claimed;
        };
        let Some(r) = comps.try_take(id) else {
            return claimed;
        };
        if let FiberState::CapParked { woken, .. } = &mut fibers[fi] {
            *woken = Some(r);
        }
        claimed = true;
    }
}

/// The root activation's id in a vCPU's resume chain (it has no fiber handle).
const ROOT_FIBER: usize = usize::MAX;

/// One vCPU's continuation: its active `Vm` and its resume `chain`. A `thread.spawn` creates a fresh
/// `VTask`; the scheduler runs them cooperatively over one shared `Mem` (single-threaded, so shared
/// memory is sequentially consistent — the determinate programs the oracle uses give the same result
/// on any correct schedule). The §12 **fiber registry is run-shared** (one handle namespace per
/// domain, held by [`drive`]), so a fiber created/suspended on one vCPU can be resumed on another
/// (D57 migration) — only the resume `chain` (the ancestor stack) is per-vCPU.
struct VTask {
    active: Vm,
    /// `ROOT_FIBER` or the handle of the fiber currently running in this vCPU.
    active_id: usize,
    /// Parked resumers: `(fiber id, its Vm, the `cont.resume` result slot awaiting (status, value))`.
    chain: Vec<(usize, Vm, u32)>,
    /// DURABILITY.md §12.8 (D-fiber-cont option A): the root computation's (context 0's) saved durable
    /// shadow-stack pointer, swapped with the in-window active word ([`super::SHADOW_SP_OFF`]) on each
    /// fiber switch so a freeze poll spills into the *running* context's region. Only meaningful on a
    /// durable run; `super::SHADOW_BASE` (context 0's region base) otherwise.
    root_shadow_sp: u64,
    /// Debug **step-into** of a §14 coroutine body — set for every *debug-engine* task (`true` by
    /// default: the single-vCPU [`DebugRun`] *and* the multi-vCPU [`ScheduledDebugRun`], where the
    /// coroutine's vCPU is pinned across the body so the op-by-op stepping stays atomic w.r.t. other
    /// vCPUs). When set, a `resume` defers the child to op-by-op stepping via
    /// the §22 invoked unit instead of running it opaquely to its next return.
    /// Only ever read by [`debug_advance_fiber`] (the debug driver); production's `step_vcpu` ignores it,
    /// so a production `VTask` carrying `true` is inert.
    /// Step-into of a §22 `Jit.invoke`d unit (single-vCPU [`DebugRun`] only;
    /// the scheduled engine keeps invoke an opaque leaf, as it does coroutines). `None` when not stepping
    /// inside an invoke. Mutually exclusive with `active_coro` — an invoked unit is seam-free (a coroutine
    /// child holds no `Jit` cap, and a `cont.*`/`spawn` inside an invoked unit `CapFault`s).
    active_invoke: Option<Box<InvokeStep>>,
    /// Step *into* a §22 `Jit.invoke`d unit rather than running it as an opaque leaf. Set **only** on the
    /// single-vCPU [`DebugRun`] (where step-into semantics live); `false` on every scheduled task, so the
    /// [`ScheduledDebugRun`] keeps invoke a leaf and never arms `active_invoke` — the scheduled pinning /
    /// snapshot paths therefore need no invoke handling. Read only by [`debug_advance_fiber`]'s JitInvoke
    /// arm; production ignores it.
    invoke_step_into: bool,
}

/// While the single-vCPU [`DebugRun`] steps *inside* a §22 `Jit.invoke`d unit, the active continuation
/// is the invoked unit's [`Vm`], not [`VTask::active`]. Unlike a coroutine child (a confined domain with
/// its own `mem`/`host`/`table`), an invoked unit is a **seam-free leaf over the caller's** window /
/// powerbox / dispatch table, so only its `Vm` is held here — the reader resolves its frames against the
/// session `mem`/`source`/`table` (module ≥ 1 for its own funcs, dispatching installed units through the
/// shared table like the production `run_invoke`). On completion the unit's returns marshal into the
/// caller's `dst` slots through the i64-slot ABI; `parent_depth` is the caller's call depth at the
/// invoke, so the stepping predicate sees a *cumulative* depth across the boundary (step-over of the
/// `invoke` runs the unit to completion), exactly like [`VTask::active_coro`].
struct InvokeStep {
    vm: Vm,
    dst: u32,
    results: Box<[ValType]>,
    parent_depth: usize,
}

impl VTask {
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<VTask, Trap> {
        Ok(VTask {
            active: Vm::new(c, entry, args)?,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            root_shadow_sp: super::SHADOW_BASE,
            active_invoke: None,
            invoke_step_into: false, // DebugRun::new_with_host flips this on for the single-vCPU engine
        })
    }

    /// The continuation the single-vCPU debugger is currently stepping: a §22 invoked unit's `Vm`
    /// (which shares the caller's window/table — the reader resolves its frames against the session
    /// `mem`/`source`; its module-≥1 SSA metadata is not plumbed, but its `IrPc`s — hence
    /// breakpoints, stepping, and backtrace — resolve via `source`) or, normally, `active`.
    fn debug_active(&self) -> &Vm {
        match &self.active_invoke {
            Some(iv) => &iv.vm,
            None => &self.active,
        }
    }
}

/// Re-point the durable active shadow-SP word from the outgoing context's region to the incoming
/// one's, on a fiber switch (DURABILITY.md §12.8, D-fiber-cont option A) — the bytecode-engine mirror
/// of the tree-walker's `shadow_switch`. The running context's live SP is the in-window word the
/// instrumented IR maintains; each *non-running* context's SP lives host-side (the root's in
/// `VTask::root_shadow_sp`, a fiber's in `fiber_sp[slot]`). A no-op unless the run is `durable` with a
/// window. `ctx` is `ROOT_FIBER` for the root or a fiber's registry slot.
fn shadow_switch(
    mem: &mut Option<Mem>,
    fiber_sp: &mut [u64],
    root_shadow_sp: &mut u64,
    durable: bool,
    out_ctx: usize,
    in_ctx: usize,
) {
    if !durable {
        return;
    }
    let Some(m) = mem.as_mut() else { return };
    // §12.8 4A.5: each context's SP word lives in its own region (root = context 0, fiber slot `s` =
    // context `s + 1`). (This bytecode durable path is unreachable today — durable hosts always run on
    // the tree-walker — but kept correct and compiling.)
    let region_of =
        |ctx: usize| super::shadow_region_base(if ctx == ROOT_FIBER { 0 } else { ctx + 1 });
    let sp = m.durable_get_sp(region_of(out_ctx));
    if out_ctx == ROOT_FIBER {
        *root_shadow_sp = sp;
    } else {
        fiber_sp[out_ctx] = sp;
    }
    let in_sp = if in_ctx == ROOT_FIBER {
        *root_shadow_sp
    } else {
        fiber_sp[in_ctx]
    };
    m.durable_set_sp(region_of(in_ctx), in_sp);
}

/// **Freeze driver** (DURABILITY.md §12.8 slice 3.1.4) — the bytecode mirror of the tree-walker's
/// `VCpu::freeze_drive`. Called once the root has run to completion under `UNWINDING` (its native
/// stack drained into context 0's shadow region): flatten every still-**parked** fiber into *its own*
/// region so the window snapshot captures it, and return the host-side residue (a [`FrozenFiber`] per
/// flattened fiber) the snapshot records and a thaw re-seeds.
///
/// Each parked fiber is resumed under `UNWINDING` like a standalone root run — a fresh single-frame
/// [`VTask`] whose active `Vm` is the parked continuation with `active_id == ROOT_FIBER` (so its
/// base-frame return ends the sub-run), the active shadow-SP pointed at the fiber's region base, and a
/// placeholder resume value delivered (mimicking `cont.resume`, so the post-suspend continuation is
/// well-formed). The transform places the poll **immediately** after the `suspend`, so the poll fires
/// before any guest code runs: the fiber unwinds with **zero forward progress** and returns. Its
/// flattened shadow-SP extent is saved (into `fiber_sp`, for the snapshot) and recorded in the
/// `FrozenFiber`. The active shadow-SP is left at the **root's** region on return, so the captured
/// window is thaw-ready (the root rewinds first; each fiber's own SP travels in its `FrozenFiber`).
///
/// `generation` is always 0: the bytecode engine is cooperative single-threaded and never recycles a
/// fiber slot, so handles equal slots (matching a non-recycled tree-walker run).
fn freeze_drive(
    fibers: &mut Vec<FiberState>,
    fiber_sp: &mut Vec<u64>,
    fiber_meta: &mut Vec<(i32, i64)>,
    dom: &Domain,
    ctx: &mut RunCtx,
    budget: u64,
) -> Result<Vec<super::FrozenFiber>, Trap> {
    // The root's post-unwind SP (context 0); restored at the end so the window is thaw-ready.
    let root_word = super::shadow_region_base(0);
    let root_sp = ctx
        .mem
        .as_ref()
        .map(|m| m.durable_get_sp(root_word))
        .unwrap_or(super::SHADOW_BASE + super::REGION_HEADER_LEN);
    let mut frozen = Vec::new();
    // Flatten parked fibers in ascending slot order, so the residue's handle namespace is dense from 0
    // (matching the tree-walker's `take_parked_for_freeze`, which always takes the lowest parked slot).
    for slot in 0..fibers.len() {
        let (vm, suspend_dst) = match std::mem::replace(&mut fibers[slot], FiberState::Done) {
            FiberState::Parked { vm, suspend_dst } => (vm, suspend_dst),
            other => {
                fibers[slot] = other; // not parked (Pending / Running / Done): nothing to flatten
                continue;
            }
        };
        let (func, sp) = fiber_meta.get(slot).copied().unwrap_or((0, 0));
        // Point the active shadow-SP at this fiber's region base (an empty shadow stack to unwind into).
        if let Some(m) = ctx.mem.as_mut() {
            m.durable_set_sp(
                super::shadow_region_base(slot + 1),
                super::shadow_region_base(slot + 1) + super::REGION_HEADER_LEN,
            );
        }
        // Deliver a placeholder resume value (inert; the thaw redelivers), then drive the fiber to its
        // base return under `UNWINDING` (zero forward progress: the poll fires immediately after the
        // suspend). `step_vcpu` runs the active `Vm` to completion in one call, and the unwind does no
        // fiber/thread ops, so the run-shared registries are untouched and the only stop is `Done`.
        let mut vm = vm;
        vm.set(suspend_dst, Reg::from_i64(0));
        let mut sub = VTask {
            active: vm,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            root_shadow_sp: root_sp,
            active_invoke: None,
            invoke_step_into: false,
        };
        match step_vcpu(
            &mut sub, fibers, fiber_sp, fiber_meta, dom, ctx, budget, false,
        )? {
            VcpuStop::Done(_) => {}
            _ => return Err(Trap::FiberFault), // a freeze unwind never spawns / instantiates / blocks
        }
        let shadow_sp = ctx
            .mem
            .as_ref()
            .map(|m| m.durable_get_sp(super::shadow_region_base(slot + 1)))
            .unwrap_or(super::SHADOW_BASE + super::REGION_HEADER_LEN);
        fiber_sp[slot] = shadow_sp;
        frozen.push(super::FrozenFiber {
            slot,
            func,
            sp,
            shadow_sp,
            generation: 0,
        });
    }
    // Leave the active shadow-SP at the root's region: the root rewinds first on thaw.
    if let Some(m) = ctx.mem.as_mut() {
        m.durable_set_sp(root_word, root_sp);
    }
    Ok(frozen)
}

/// Scan every live activation of `vm`'s continuation — the active window plus each suspended caller
/// on the call stack — for §GC `gc.roots` candidate words, feeding each 64-bit half (`lo`/`hi`, so a
/// `v128` contributes both) to `consider`. Each activation occupies `regs[base .. base + nslots)` of
/// the function-wide register file (the window model), so this covers exactly that function's live
/// slots — a **sound superset** of the tree-walker's per-block `frame.vals` (it also retains
/// already-dead values from other blocks of the same function, a conservative over-approximation, as
/// the JIT's native-stack scan does — the backends legitimately differ, GC.md §3.2). The register
/// file only ever holds guest words (or default `0`), so `consider`'s mask+range filter keeps any
/// host data out by construction.
fn scan_vm_roots(vm: &Vm, source: &ModuleSource, consider: &mut impl FnMut(u64)) {
    let frames = std::iter::once((vm.module, vm.cur, vm.base))
        .chain(vm.stack.iter().map(|&(m, p, b, _, _)| (m, p, b)));
    for (module, prog, base) in frames {
        let Some(c) = source.get(module) else {
            continue;
        };
        let n = c.progs[prog].nslots as usize;
        let end = (base + n).min(vm.regs.len());
        for r in &vm.regs[base..end] {
            consider(r.lo);
            consider(r.hi);
        }
    }
}

/// Emit a §GC `gc.roots` result: write the first `cap` roots (ascending, already deduplicated by the
/// `BTreeSet`) as little-endian `i64`s into guest memory at `buf` — reusing the confined buffer-write
/// path (a forged/unmapped/RO buffer is a `MemoryFault`) — and return the **total** found.
fn gc_write(
    mem: &mut Option<Mem>,
    buf: u64,
    cap: usize,
    roots: std::collections::BTreeSet<u64>,
) -> Result<i64, Trap> {
    let total = roots.len() as i64;
    let mut bytes = Vec::with_capacity(roots.len().min(cap) * 8);
    for w in roots.into_iter().take(cap) {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.as_mut()
        .ok_or(Trap::Malformed)?
        .write_bytes_impl(buf, &bytes)
        .ok_or(Trap::MemoryFault)?;
    Ok(total)
}

/// Run an invoked §22 unit (`Jit.invoke`) synchronously: a fresh `Vm` for `module`'s entry (func 0)
/// over the shared window/powerbox and the **shared** dispatch table (so the unit's `call_indirect`
/// reaches installed units), to completion. An invoked unit is threads-/seam-free — spawning,
/// event-parking, or re-installing `CapFault`s — but it **may host fibers** (DESIGN.md §22
/// "Concurrency", renegotiated 2026-07-30): `cont.*`/`suspend` are serviced here against an
/// **invoke-confined** registry (a fiber lives and dies within this one invoke — no migration to
/// the run's fibers), with entries resolved through module 0's natural table exactly as
/// [`step_vcpu`]'s arms do (a fiber over an *installed* unit function is the same deferred case).
/// A `suspend` at the invoke root would park the synchronous invoke — `CapFault`, the seam-free
/// half of the contract ("a unit runs its own scheduler to completion"). No durability shadowing:
/// a freeze never lands mid-invoke (snapshot paths carry no invoke state), so unlike `step_vcpu`
/// there is no `fiber_sp`/`shadow_switch` bookkeeping. A trap propagates to the invoker.
fn run_invoke(
    source: &ModuleSource,
    table: &SharedSlots,
    module: usize,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
) -> Result<Vec<Value>, Trap> {
    let unit = source.get(module).ok_or(Trap::Malformed)?;
    let mut active = Vm::new(&unit, 0, args)?;
    active.module = module;
    // The interpreted invoke completes synchronously, so its fiber registry is loop-local; the
    // emitted-invoke bounce path threads the vCPU's persistent registry instead (`bounce_call`).
    drive_nested(
        source,
        table,
        active,
        fuel,
        mem,
        host,
        &mut Vec::new(),
        None,
    )
}

/// The shared nested-drive loop under [`run_invoke`] (an interpreted `Jit.invoke`) and
/// [`Vcpu::bounce_call`] (#846 — one cross-tier callback out of an *emitted* unit): drive `active`
/// to completion over the shared window/powerbox/dispatch-table, servicing fibers against `fibers`.
/// The registry is caller-owned so the bounce path can persist it across the several bounces of one
/// emitted invoke (a fiber parked by one callback is resumable by a later one — exactly the
/// one-registry-per-invoke scope the interpreted loop has by construction).
/// The run-level parallel-array halves `drive_nested` mirrors in run-registry mode (#880): the
/// fibers' durable shadow-SPs and their `(entry func, sp)` freeze metadata.
type RunFiberMeta<'a> = (&'a mut Vec<u64>, &'a mut Vec<(i32, i64)>);

#[allow(clippy::too_many_arguments)] // the nested-drive seam: window + registry halves, all borrowed
fn drive_nested(
    source: &ModuleSource,
    table: &SharedSlots,
    mut active: Vm,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
    fibers: &mut Vec<FiberState>,
    // #880 — `Some((fiber_sp, fiber_meta))` when `fibers` is the vCPU's **run-level** registry (a
    // bounce out of a TIERUP region: the callback's fibers must persist for the run to resume
    // later, exactly as the same call inline would register them). `ContNew` then mirrors
    // `step_vcpu`'s parallel-array pushes so the run's bookkeeping stays index-aligned; the
    // durability `shadow_switch` is deliberately absent — a bounce host is never a durable run
    // (the pump), and the interpreted invoke path passes `None` (invoke-confined registry).
    mut run_meta: Option<RunFiberMeta<'_>>,
) -> Result<Vec<Value>, Trap> {
    // The resumer chain (`(resumer's fiber id, resumer, dst)`). Invariant: `chain` is non-empty
    // iff `active` is a fiber (`active_id` then indexes `fibers`).
    let mut chain: Vec<(usize, Vm, u32)> = Vec::new();
    let mut active_id = usize::MAX; // sentinel while the root frame is active
    loop {
        match active.resume(source, table, fuel, mem, host, u64::MAX)? {
            Outcome::Done(vals) => match chain.pop() {
                // The unit entry finished — the invoke's results.
                None => return Ok(vals),
                // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` back.
                Some((rid, resumer, rdst)) => {
                    fibers[active_id] = FiberState::Done;
                    let retval = vals.first().copied().unwrap_or(Value::I64(0));
                    active = resumer;
                    active_id = rid;
                    active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                    active.set(rdst + 1, Reg::from_value(retval));
                }
            },
            Outcome::Suspended => {}
            // F2 — a punted host call inside an invoked unit keeps the pre-F2 inline wait
            // (the unit is a seam-free atomic leaf, DESIGN §22: no park surface exists here,
            // so blocking in place is the contract, not a divergence). Because every cap call
            // completes inline, an invoke fiber is never `CapParked` — no drain arm needed.
            Outcome::CapPending { id, dst } => {
                let r = host.with(|p| p.completions()).wait(id);
                active.set(dst, Reg::from_i64(r));
            }
            Outcome::ContNew { funcref, sp, dst } => {
                if fibers.len() + 1 >= super::MAX_FIBERS {
                    return Err(Trap::FiberFault);
                }
                let h = fibers.len() as i32;
                fibers.push(FiberState::Pending { funcref, sp });
                // Run-registry mode (#880): keep the parallel arrays index-aligned with the run's
                // (`step_vcpu`'s ContNew arm, minus the durable shadow bookkeeping — see the
                // `run_meta` doc above).
                if let Some((fiber_sp, fiber_meta)) = run_meta.as_mut() {
                    fiber_sp
                        .push(super::shadow_region_base(h as usize + 1) + super::REGION_HEADER_LEN);
                    let func_idx = (funcref as u32 as usize & source.primary().table_mask) as i32;
                    fiber_meta.push((func_idx, sp));
                }
                active.set(dst, Reg::from_i32(h));
            }
            // Fibers here never event-park (every park surface faults or waits inline above), so
            // the I48 `blocking` flag is a no-op — `cont.resume.block` behaves like `cont.resume`.
            Outcome::ContResume {
                kh,
                arg,
                dst,
                blocking: _,
                resume_ip: _,
            } => {
                let k = kh as usize;
                let target = match fibers.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let (funcref, sp) = match std::mem::replace(
                            slot,
                            FiberState::Running { blocking_ip: None },
                        ) {
                            FiberState::Pending { funcref, sp } => (funcref, sp),
                            _ => unreachable!(),
                        };
                        // Resolve through module 0's natural table + the fiber signature, exactly
                        // as `step_vcpu` (the raw-slot naming of DESIGN.md §22's renegotiation).
                        let m0 = source.primary();
                        let f = (funcref as u32 as usize) & m0.table_mask;
                        let ok = m0
                            .sigs
                            .get(f)
                            .is_some_and(|(p, r)| p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS);
                        if !ok {
                            return Err(Trap::FiberFault);
                        }
                        Vm::new(&m0, f, &[Value::I64(sp), Value::I64(arg)])?
                    }
                    Some(slot @ FiberState::Parked { .. }) => {
                        match std::mem::replace(slot, FiberState::Running { blocking_ip: None }) {
                            FiberState::Parked {
                                mut vm,
                                suspend_dst,
                            } => {
                                vm.set(suspend_dst, Reg::from_i64(arg));
                                vm
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => return Err(Trap::FiberFault), // forged / Running / Done
                };
                let resumer = std::mem::replace(&mut active, target);
                chain.push((active_id, resumer, dst));
                active_id = k;
            }
            Outcome::FiberSuspend { value, dst } => {
                // An empty chain means the unit entry itself tried to `suspend` — that would park
                // the synchronous invoke, the seam the §22 contract forbids.
                let Some((rid, resumer, rdst)) = chain.pop() else {
                    return Err(Trap::CapFault);
                };
                let suspended = std::mem::replace(&mut active, resumer);
                fibers[active_id] = FiberState::Parked {
                    vm: suspended,
                    suspend_dst: dst,
                };
                active_id = rid;
                active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
                active.set(rdst + 1, Reg::from_i64(value));
            }
            _ => return Err(Trap::CapFault),
        }
    }
}

/// Why [`step_vcpu`] returned control to the scheduler: the vCPU finished, or it hit a multi-vCPU
/// (`thread.*` / `memory.*`) event the scheduler must service. Intra-vCPU fiber switches never reach
/// here — `step_vcpu` handles them against the vCPU's own registry.
enum VcpuStop {
    /// I48 — a `cont.resume.block` whose target fiber is still event-parked: idle this task on the
    /// fiber (`TaskState::BlockedOnFiber`). `step_vcpu` already rewound the resumer's cursor to the
    /// resume op, so the wake re-executes it.
    BlockOnFiber {
        fiber: usize,
    },
    /// §3.6 (I36 slice 2): park this task on a live-call `ticket` against `callee` (see
    /// [`Outcome::LiveCall`] — the enqueue already happened in the op exec).
    LiveCall {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// F2 (FIBER_PARK.md): a punted offloadable dispatch with an exactly-`i64` reply
    /// ([`Outcome::CapPending`]) — the cooperative driver fiber-parks a punting fiber; every
    /// other driver waits inline on the completion (the I45 posture).
    CapPending {
        id: u64,
        dst: u32,
    },
    /// §3.6 (I36 slice 2): park this task in `svc.wait` on its own domain ([`Outcome::SvcWait`]).
    SvcWait,
    /// §3.6 (I36 slice 2): mint a live offer over child `child`'s export ([`Outcome::ChildOffer`]).
    ChildOffer {
        child: i32,
        export: u32,
        dst: u32,
    },
    /// FORK.md §9.2 — `clone_caller`: fork the caller parked on this handler's dispatch into a twin
    /// ([`Outcome::CloneCaller`]). The driver reads the running handler's `serve_ticket` to name it.
    CloneCaller {
        reply_orig: Option<i64>,
        reply_twin: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap`: reap twin `pid` on behalf of the caller parked on this handler's
    /// dispatch ([`Outcome::Reap`]).
    Reap {
        pid: i64,
        dst: u32,
        has_result: bool,
    },
    Done(Vec<Value>),
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): run the emitted `f{func}` region on the
    /// host, delivering its `n_results` results to absolute slot `dst` via `deliver_tierup`.
    /// `mapped` is the entry-snapshot scalar committed extent the host must write to the emitted
    /// `"mapped"` global first (#717 — see [`Outcome::TierUp`]).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
        mapped: u64,
    },
    Spawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
        /// The spawning frame's module — `func` resolves there (CONSOLIDATION.md §11).
        module: u32,
    },
    Join {
        handle: i32,
        dst: u32,
    },
    /// §14 `Instantiator.instantiate` — the driver (which owns the task set / extra environments)
    /// builds the confined executor child and registers it as a joinable thread.
    Instantiate {
        ibase: u64,
        isize: u64,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        dst: u32,
        /// op 11 `instantiate_named`: resolved `(grants_ptr, grants_n)` window coordinates of the
        /// child's by-name grant list (op 0 is `None`).
        grants: Option<(u64, u64)>,
        /// §3d (op 17): the record's `Budget` handle (`0` = none) — funded at the driver's commit.
        budget: i32,
    },
    /// §14 `Instantiator.instantiate_module` — the driver additionally resolves + compiles the
    /// host-granted `Module` (`mh`) and runs it as the confined child's program.
    InstantiateModule {
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        dst: u32,
        /// op 13 `instantiate_module_named`: resolved `(grants_ptr, grants_n)` window coordinates of
        /// the child's by-name grant list (op 5 is `None`).
        grants: Option<(u64, u64)>,
        /// §3d (op 17): the record's `Budget` handle (`0` = none) — see above.
        budget: i32,
    },
    Wait {
        base: u64,
        expected: u64,
        width: u32,
        timeout: u64,
        dst: u32,
    },
    Notify {
        base: u64,
        count: i32,
        dst: u32,
    },
    /// §22 `Jit.install` — the driver (which owns the mutable `Domain`) compiles + installs the unit.
    JitInstall {
        h: i32,
        code: i32,
        dst: u32,
    },
    /// §22 `Jit.uninstall` — the driver clears the table slot.
    JitUninstall {
        h: i32,
        slot: i64,
        dst: u32,
    },
    /// §22 `Jit.invoke` — the driver runs the unit synchronously over the shared window.
    JitInvoke {
        h: i32,
        code: i32,
        argv: Box<[i64]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// Blocking-stdin park (see [`Outcome::StdinPark`]) — the `Vcpu` driver surfaces it as
    /// [`VcpuEvent::StdinPark`]; no residue, since the read re-issues on resume.
    StdinPark,
}

/// How the eval loop reaches the powerbox (THREADS.md 4c-host). The cooperative `drive` owns the host
/// exclusively (`&mut Host`); the **parallel** driver shares one `Arc<Mutex<Host>>` across vCPU threads
/// and takes the lock only for the duration of a single `cap.call` — so compute/atomics/futex between
/// calls stay lock-free (genuine parallelism), exactly the tree-walker's model. Determinism is *not*
/// lost: cooperative is uncontended and dispatches in the same fixed order as before (the oracle);
/// parallel is the opt-in mode whose stateful-cap interleaving races, as real threads do.
enum HostCell<'a> {
    /// Single-owner exclusive access — the cooperative `drive`, the debugger, coroutines, §14 children.
    Excl(&'a mut Host),
    /// Shared behind a lock — the parallel driver's vCPUs; `with` takes the lock per host call.
    Shared(&'a std::sync::Mutex<Host>),
}

impl HostCell<'_> {
    /// Run `f` with exclusive access to the powerbox: directly (`Excl`) or under a brief lock
    /// (`Shared`). `f`'s result is owned (no borrow escapes the lock), so the lock is held only across
    /// the one host call.
    #[inline]
    fn with<R>(&mut self, f: impl FnOnce(&mut Host) -> R) -> R {
        match self {
            HostCell::Excl(h) => f(h),
            HostCell::Shared(m) => f(&mut m.lock_unpoisoned()),
        }
    }
}

/// The per-vCPU execution environment a [`step_vcpu`] runs against: the dispatch `table` it uses
/// (the shared domain table, or a §14 confined child's own natural table), its `fuel` budget, its
/// linear `mem`, and its capability `host`. The root vCPU and its `thread.spawn` siblings share the
/// domain's (env `None`); a §14 `instantiate` child carries its own confined [`ChildEnv`]. Bundled so
/// [`step_vcpu`] takes one ref instead of four (and so the per-task selection has a single type).
struct RunCtx<'a> {
    table: &'a SharedSlots,
    fuel: &'a mut u64,
    mem: &'a mut Option<Mem>,
    host: HostCell<'a>,
    /// DURABILITY.md §12.8: the domain is durable, so each fiber switch maintains the per-context
    /// shadow-SP word ([`shadow_switch`]). Read once from `Host::is_durable` by [`drive`].
    durable: bool,
}

/// Run one vCPU (its active `Vm` and any fibers it switches among) until it finishes or hits a
/// multi-vCPU event. Fiber `Outcome`s are serviced here exactly as `run_inner`'s `cont.*` arms switch
/// the active frame stack; `thread.*`/`memory.*` `Outcome`s are handed up to [`drive`]. `budget` only
/// slices *where* the active `Vm` pauses (Slice 1c-2); it never changes results.
#[allow(clippy::too_many_arguments)] // scheduler seam: the vCPU state, registry, domain + the I48 cooperative flag
fn step_vcpu(
    vt: &mut VTask,
    fibers: &mut Vec<FiberState>,
    fiber_sp: &mut Vec<u64>,
    fiber_meta: &mut Vec<(i32, i64)>,
    dom: &Domain,
    ctx: &mut RunCtx,
    budget: u64,
    // I48: only the cooperative `drive` scheduler can idle a blocking `cont.resume.block` (park the
    // resumer's task via `VcpuStop::BlockOnFiber`). The OS-thread parallel paths pass `false` and
    // take the advisory `FIBER_PARKED` poll instead — their idle is the follow-up slice (the same
    // OS-thread-block problem as the Cranelift JIT).
    cooperative: bool,
) -> Result<VcpuStop, Trap> {
    loop {
        match vt.active.resume(
            &dom.source,
            ctx.table,
            &mut *ctx.fuel,
            &mut *ctx.mem,
            &mut ctx.host,
            budget,
        )? {
            // Budget exhausted (sliced harness only): re-enter the same activation; its cursor is
            // already persisted, so this is transparent.
            Outcome::Suspended => {}
            Outcome::Done(vals) => match vt.chain.pop() {
                // The vCPU's root activation finished.
                None => return Ok(VcpuStop::Done(vals)),
                // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` to its resumer.
                Some((rid, resumer, rdst)) => {
                    fibers[vt.active_id] = FiberState::Done;
                    // Fiber switch (returning fiber → its resumer): re-point the durable shadow-SP.
                    shadow_switch(
                        ctx.mem,
                        fiber_sp,
                        &mut vt.root_shadow_sp,
                        ctx.durable,
                        vt.active_id,
                        rid,
                    );
                    let retval = vals.first().copied().unwrap_or(Value::I64(0));
                    vt.active = resumer;
                    vt.active_id = rid;
                    vt.active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                    vt.active.set(rdst + 1, Reg::from_value(retval));
                }
            },
            Outcome::ContNew { funcref, sp, dst } => {
                if fibers.len() + 1 >= super::MAX_FIBERS {
                    return Err(Trap::FiberFault);
                }
                let h = fibers.len() as i32;
                fibers.push(FiberState::Pending { funcref, sp });
                // A fresh fiber (registry slot `h`) is shadow context `h + 1`; its saved shadow-SP
                // starts at its region base (empty shadow stack) — so a later switch into it points
                // the active word there (DURABILITY.md §12.8).
                fiber_sp.push(super::shadow_region_base(h as usize + 1) + super::REGION_HEADER_LEN); // §12.8 4A.5: empty = frame base (past the in-region SP + thaw words)
                                                                                                     // Freeze residue (DURABILITY.md §12.8): record the fiber's re-entry metadata — its
                                                                                                     // **resolved** entry function index (the natural-table lookup `cont.resume` does, so
                                                                                                     // a `FrozenFiber.func` matches the tree-walker's `Frame::func`) and data-stack base —
                                                                                                     // so the freeze driver can emit a `FrozenFiber` for it even after it parks.
                let func_idx = (funcref as u32 as usize & dom.source.primary().table_mask) as i32;
                fiber_meta.push((func_idx, sp));
                vt.active.set(dst, Reg::from_i32(h));
            }
            Outcome::ContResume {
                kh,
                arg,
                dst,
                blocking,
                resume_ip,
            } => {
                let k = kh as usize;
                // I48: if this resume is `cont.resume.block`, tag the switched-in fiber so a park
                // inside it idles this task (rewinding to `resume_ip`) instead of the FIBER_PARKED
                // poll. `None` for a plain `cont.resume`.
                let blocking_ip = blocking.then_some(resume_ip);
                // F2 (FIBER_PARK.md) — a poll of a cap-parked fiber runs the ordered drain
                // first (so a busy resume-poll loop observes its completion without waiting
                // for driver idle — the WaitParked `real_deadline` shape, completion form).
                // The drain, not a direct `try_take`, so a poll of a LATER id never lets its
                // ready result overtake an earlier outstanding park (the §18 pin).
                if matches!(
                    fibers.get(k),
                    Some(FiberState::CapParked { woken: None, .. })
                ) {
                    let comps = ctx.host.with(|p| p.completions());
                    drain_cap_parked(fibers, &comps);
                }
                // Claim fiber `k` from the **run-shared** registry: a pending fiber starts (call
                // `funcref(sp, arg)`), a parked one continues (the new `arg` becomes its `suspend`'s
                // result) — possibly one suspended on *another* vCPU (D57 migration). Anything else
                // (forged / already running on a vCPU / done) is inert.
                let target = match fibers.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let (funcref, sp) =
                            match std::mem::replace(slot, FiberState::Running { blocking_ip }) {
                                FiberState::Pending { funcref, sp } => (funcref, sp),
                                _ => unreachable!(),
                            };
                        // Resolve the fiber entry through module 0's natural table + `fiber_sig` —
                        // a forged/mistyped funcref is a `FiberFault`. A *submitted unit* may now
                        // create fibers (DESIGN.md §22 "Concurrency", renegotiated 2026-07-30); it
                        // names the entry by a raw slot (`cont.new <slot>`), and an entry that is an
                        // original (module-0) function resolves here exactly as the JIT's shared
                        // `fn_table` and the tree-walker's `dispatch_indirect` do. (A fiber over an
                        // *installed* unit function — a module ≥ 1 entry — is the deferred case; it
                        // would need the module-aware `DomainTable` here, as those two backends use.)
                        let m0 = dom.source.primary();
                        let f = (funcref as u32 as usize) & m0.table_mask;
                        let ok = m0
                            .sigs
                            .get(f)
                            .is_some_and(|(p, r)| p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS);
                        if !ok {
                            return Err(Trap::FiberFault);
                        }
                        let mut fvm = Vm::new(&m0, f, &[Value::I64(sp), Value::I64(arg)])?;
                        // §12.8 4A.5: this fiber spills into its own region (slot `k` = context `k + 1`).
                        fvm.durable_region_base = super::shadow_region_base(k + 1);
                        fvm
                    }
                    Some(slot @ FiberState::Parked { .. }) => {
                        match std::mem::replace(slot, FiberState::Running { blocking_ip }) {
                            FiberState::Parked {
                                mut vm,
                                suspend_dst,
                            } => {
                                vm.set(suspend_dst, Reg::from_i64(arg));
                                vm
                            }
                            _ => unreachable!(),
                        }
                    }
                    // §3.6 slice 5a: an event-parked fiber (blocked in `memory.wait`). Woken —
                    // or with its real deadline passed (the timeout fires at the poll, so a
                    // cooperative resume-poll loop terminates) — the resume delivers the wait's
                    // status into the fiber and continues it (the resume `arg` is deliberately
                    // NOT delivered, matching the oracle's `LiveWoken`); still blocked, the
                    // resumer gets `(FIBER_PARKED, 0)` without a switch (the cooperative poll).
                    Some(slot @ FiberState::WaitParked { .. }) => {
                        let FiberState::WaitParked {
                            woken,
                            real_deadline,
                            ..
                        } = slot
                        else {
                            unreachable!()
                        };
                        let fired = woken.take().or_else(|| {
                            (sched_wall_now() >= *real_deadline).then_some(super::WAIT_TIMED_OUT)
                        });
                        let Some(st) = fired else {
                            // I48: a blocking resume of a still-parked fiber idles this task on the
                            // fiber (its deadline is already in the idle scan; notify wakes it too),
                            // rewinding the resumer's cursor so the wake re-executes the resume. A
                            // plain resume returns the FIBER_PARKED poll (guest loops).
                            if blocking && cooperative {
                                vt.active.pc = resume_ip;
                                return Ok(VcpuStop::BlockOnFiber { fiber: k });
                            }
                            vt.active.set(dst, Reg::from_i32(super::FIBER_PARKED));
                            vt.active.set(dst + 1, Reg::from_i64(0));
                            continue;
                        };
                        match std::mem::replace(slot, FiberState::Running { blocking_ip }) {
                            FiberState::WaitParked {
                                mut vm, wait_dst, ..
                            } => {
                                vm.set(wait_dst, Reg::from_i32(st));
                                vm
                            }
                            _ => unreachable!(),
                        }
                    }
                    // F2 — a cap-parked fiber (blocked on its punt completion). Claimed by
                    // the drain above (`woken`), the resume delivers the scalar into the
                    // `cap.call`'s result register and continues it (the resume `arg` is
                    // deliberately NOT delivered — the oracle's `LiveWoken`); still in
                    // flight, the resumer gets `(FIBER_PARKED, 0)` without a switch.
                    Some(slot @ FiberState::CapParked { .. }) => {
                        let FiberState::CapParked { woken, .. } = slot else {
                            unreachable!()
                        };
                        let Some(r) = woken.take() else {
                            // I48: blocking resume idles this task on the cap-parked fiber; the
                            // ordered completion drain wakes it. Plain resume returns FIBER_PARKED.
                            if blocking && cooperative {
                                vt.active.pc = resume_ip;
                                return Ok(VcpuStop::BlockOnFiber { fiber: k });
                            }
                            vt.active.set(dst, Reg::from_i32(super::FIBER_PARKED));
                            vt.active.set(dst + 1, Reg::from_i64(0));
                            continue;
                        };
                        match std::mem::replace(slot, FiberState::Running { blocking_ip }) {
                            FiberState::CapParked {
                                mut vm,
                                dst: cap_dst,
                                ..
                            } => {
                                vm.set(cap_dst, Reg::from_i64(r));
                                vm
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => return Err(Trap::FiberFault), // forged / Running / Done
                };
                // Fiber switch (resumer → fiber `k`): re-point the durable shadow-SP before the swap.
                shadow_switch(
                    ctx.mem,
                    fiber_sp,
                    &mut vt.root_shadow_sp,
                    ctx.durable,
                    vt.active_id,
                    k,
                );
                let resumer = std::mem::replace(&mut vt.active, target);
                vt.chain.push((vt.active_id, resumer, dst));
                vt.active_id = k;
            }
            Outcome::FiberSuspend { value, dst } => {
                // Pop the resumer to switch back to; an empty chain means the root tried to
                // `suspend`, which is a `FiberFault` (the root has no resumer).
                let (rid, resumer, rdst) = vt.chain.pop().ok_or(Trap::FiberFault)?;
                // Fiber switch (suspending fiber → its resumer): re-point the durable shadow-SP.
                shadow_switch(
                    ctx.mem,
                    fiber_sp,
                    &mut vt.root_shadow_sp,
                    ctx.durable,
                    vt.active_id,
                    rid,
                );
                let suspended = std::mem::replace(&mut vt.active, resumer);
                fibers[vt.active_id] = FiberState::Parked {
                    vm: suspended,
                    suspend_dst: dst,
                };
                vt.active_id = rid;
                vt.active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
                vt.active.set(rdst + 1, Reg::from_i64(value));
            }
            Outcome::TierUp {
                func,
                argv,
                dst,
                results,
                mapped,
            } => {
                return Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                })
            }
            Outcome::ThreadSpawn {
                func,
                sp,
                arg,
                dst,
                module,
            } => {
                return Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module: module as u32,
                })
            }
            Outcome::ThreadJoin { handle, dst } => return Ok(VcpuStop::Join { handle, dst }),
            // §3.6 (I36 slice 2): the serve/call/offer trio surface straight to the driver. The
            // qualification veto keeps them out of fiber contexts, so no registry state is live.
            Outcome::LiveCall {
                ticket,
                callee,
                dst,
            } => {
                return Ok(VcpuStop::LiveCall {
                    ticket,
                    callee,
                    dst,
                })
            }
            Outcome::SvcWait => return Ok(VcpuStop::SvcWait),
            Outcome::ChildOffer { child, export, dst } => {
                return Ok(VcpuStop::ChildOffer { child, export, dst })
            }
            Outcome::CloneCaller {
                reply_orig,
                reply_twin,
                dst,
                has_result,
            } => {
                return Ok(VcpuStop::CloneCaller {
                    reply_orig,
                    reply_twin,
                    dst,
                    has_result,
                })
            }
            Outcome::Reap {
                pid,
                dst,
                has_result,
            } => {
                return Ok(VcpuStop::Reap {
                    pid,
                    dst,
                    has_result,
                })
            }
            Outcome::Instantiate {
                ibase,
                isize: isz,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            } => {
                return Ok(VcpuStop::Instantiate {
                    ibase,
                    isize: isz,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                })
            }
            Outcome::InstantiateModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            } => {
                return Ok(VcpuStop::InstantiateModule {
                    ibase,
                    isize: isz,
                    mh,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                })
            }
            Outcome::CapPending { id, dst } => return Ok(VcpuStop::CapPending { id, dst }),
            Outcome::MemoryWait {
                base,
                expected,
                width,
                timeout,
                dst,
            } => {
                return Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                })
            }
            // Blocking-stdin park (owned-host session): surface it for the `Vcpu` driver to pump.
            Outcome::StdinPark => return Ok(VcpuStop::StdinPark),
            Outcome::MemoryNotify { base, count, dst } => {
                return Ok(VcpuStop::Notify { base, count, dst })
            }
            Outcome::JitInstall { h, code, dst } => {
                return Ok(VcpuStop::JitInstall { h, code, dst })
            }
            Outcome::JitUninstall { h, slot, dst } => {
                return Ok(VcpuStop::JitUninstall { h, slot, dst })
            }
            Outcome::JitInvoke {
                h,
                code,
                argv,
                dst,
                params,
                results,
            } => {
                return Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                })
            }
            // §GC `gc.roots`: scan the whole vCPU continuation — the active window, its call stack
            // (covered by `scan_vm_roots`), every resume-chain ancestor, every parked fiber, and every
            // suspended coroutine — for words that (masked) land in `[lo, hi)`. A **sound superset**
            // of the genuine roots, kept in-window by the range filter (GC.md §3.2).
            Outcome::GcRoots {
                lo,
                hi,
                mask,
                buf,
                cap,
                dst,
            } => {
                let mut roots = std::collections::BTreeSet::new();
                {
                    let mut consider = |w: u64| {
                        let m = w & mask;
                        if m >= lo && m < hi {
                            roots.insert(m);
                        }
                    };
                    scan_vm_roots(&vt.active, &dom.source, &mut consider);
                    for (_, vm, _) in &vt.chain {
                        scan_vm_roots(vm, &dom.source, &mut consider);
                    }
                    for fib in fibers.iter() {
                        // §3.6 slice 5a / F2: an event-parked fiber (`WaitParked` futex,
                        // `CapParked` punt completion) holds live frames exactly like a
                        // suspended one — scan all three, or a root held across a fiber's
                        // blocking point would be missed (unsound for GC.md §3.2).
                        if let FiberState::Parked { vm, .. }
                        | FiberState::WaitParked { vm, .. }
                        | FiberState::CapParked { vm, .. } = fib
                        {
                            scan_vm_roots(vm, &dom.source, &mut consider);
                        }
                    }
                }
                let total = gc_write(ctx.mem, buf, cap, roots)?;
                vt.active.set(dst, Reg::from_i64(total));
            }
        }
    }
}

/// `fiber_sig` params/results, inlined so the driver can compare without allocating a `FuncType`.
const FIBER_PARAMS: [ValType; 2] = [ValType::I64, ValType::I64];
const FIBER_RESULTS: [ValType; 1] = [ValType::I64];

/// A §14 `instantiate` child's confined runtime, owned by [`drive`] alongside the task set. Its `mem`
/// is a `nested_view` sub-window sharing the parent's backing (the §14 shared data plane), its `host`
/// an attenuated powerbox (an `Instantiator` + an `AddressSpace`, each over `[0, child_size)`), its
/// `table` a fresh **natural** dispatch table over module 0 (no access to installed §22 units — like
/// the tree-walker's fresh `DomainTable::new(&cfuncs, 0)`), and `fuel` a sub-allocated quota.
struct ChildEnv {
    mem: Option<Mem>,
    /// The child's live powerbox. `Arc<Mutex<…>>` (single-threaded here, so uncontended) so a
    /// §3.6 live-callee offer can hold the SAME callee the tree-walker's `wire_live_impl`
    /// machinery expects — enqueue, offer-shape, and settle all go through the shared type.
    host: std::sync::Arc<std::sync::Mutex<Host>>,
    table: SharedSlots,
    fuel: u64,
}

/// A scheduled vCPU and its blocking state.
struct TaskSlot {
    vt: VTask,
    /// This vCPU's `thread.spawn` / `instantiate` children (handle = index → global task index).
    /// `None` = joined. (Both seams share one handle namespace, matching the tree-walker's `threads`.)
    threads: Vec<Option<usize>>,
    /// The runtime environment this vCPU steps against: `None` = the shared domain (root + its
    /// `thread.spawn` siblings); `Some(k)` = the confined `extra_envs[k]` of a §14 `instantiate` child
    /// (and any threads it spawns, which share its window — they inherit the same env index).
    env: Option<usize>,
    state: TaskState,
}

enum TaskState {
    Runnable,
    /// Parked on `thread.join` of task `child`; deliver its result to `dst` and wake.
    BlockedJoin {
        child: usize,
        slot: usize,
        dst: u32,
    },
    /// Parked on `memory.wait` at futex `key` until notified or `deadline` (logical clock). The key is
    /// **backing-identity canonical** ([`super::FutexKey`]): two confined `instantiate` children that
    /// mapped the same `SharedRegion` into their separate windows park/wake on the same key (S1c), so a
    /// pipe ring between concurrent stages rendezvous. A plain `thread.spawn` sibling (shared root
    /// window, anonymous page) keys on its confined address — `FutexKey::Anon`, as before.
    BlockedWait {
        key: super::FutexKey,
        deadline: u64,
        dst: u32,
    },
    /// §3.6 (I36 slice 2): parked in `svc.wait` on this task's own domain (its env's host); a
    /// caller's enqueue on that host re-admits it (the rewound op re-executes the drain).
    BlockedSvc,
    /// §3.6 (I36 slice 2): parked on a live-call `ticket` against `callee`'s completion cells;
    /// the settle-wake scan delivers the reply to `dst` (the claim — the tree-walker's
    /// `cap_reply` preference, cooperative form).
    BlockedTicket {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// FORK.md §9.2 — parked in `reap` (`wait(pid)`) until fork twin `pid` (a task index) finishes;
    /// the settle scan delivers its exit status ([`super::reap_status`]) to `dst` and wakes. A
    /// trapped twin reaps as a nonzero crash status, never a propagated trap (reap ≠ join).
    BlockedReap {
        pid: usize,
        dst: u32,
    },
    /// I48 — parked in a blocking `cont.resume.block` on fiber `fiber` (event-parked, not yet woken).
    /// The resumer's cursor was rewound to the resume op; when `fiber` is woken (idle-timer, notify,
    /// or the cap-completion drain) this task is marked `Runnable` and re-executes the resume, which
    /// now claims the woken fiber and switches in. Burns no fuel while parked (skipped by the runnable
    /// scan) — the idle-not-spin proof.
    BlockedOnFiber {
        fiber: usize,
    },
    /// Finished — its result (or trap) is retained for a joiner.
    Done(Result<Vec<Value>, Trap>),
}

/// Drive a whole domain — the entry vCPU plus any `thread.spawn` children — to completion on a
/// **cooperative single-threaded scheduler** sharing one `Mem`. The oracle's concurrent programs are
/// interleaving-invariant (verified by the tree-walker via stress / seed-sweep / DPOR), so any
/// correct schedule yields the same result; a deterministic lowest-index-first pick keeps it
/// reproducible. Blocking (`join` / `wait`) parks a task; `notify` / child completion wakes it; a
/// stuck set advances a logical clock to the next `wait` deadline (or deadlocks → `ThreadFault`,
/// matching the deterministic explorer). The run ends when the **root** vCPU completes.
fn drive(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    budget: u64,
) -> Result<Vec<Value>, Trap> {
    // The native driver never enables tier-up (no eligibility bitmap), so `pump` runs the whole
    // schedule and returns `Done`; a `TierUp` yield is impossible here.
    let mut sched = CoopSched::new(&dom, entry, args, fuel, mem, host, None)?;
    match sched.pump(&dom, mem, host, fuel, budget)? {
        CoopStep::Done(vals) => Ok(vals),
        CoopStep::TierUp { .. } => unreachable!("tier-up not enabled on the native driver"),
        CoopStep::JitInvoke { .. } => {
            unreachable!("Jit.invoke surfacing not enabled on the native driver")
        }
    }
}

/// A pause point of [`CoopSched::pump`]: either the run finished (`Done`) or a module-0 task hit an
/// eligible direct `Call` and the emitted region must run on the host (`TierUp`) before the paused
/// task is resumed via [`CoopSched::deliver_tierup`]. `pump` returns `Err(trap)` for a run-fatal trap
/// (the root task trapped, or a driver operation failed) — mirroring the `Result` `drive` returns.
enum CoopStep {
    /// The root task returned; these are the run's results.
    Done(Vec<Value>),
    /// A task paused on an eligible module-0 `Call` to `func` with raw i64 arg slots `argv`; `mapped`
    /// is the window's committed scalar extent for the emitted `"mapped"` global (#717 host sync).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        mapped: u64,
    },
    /// A task paused on a §22 `Jit.invoke` of a runtime-compiled unit that has **emitted wasm**, an
    /// all-scalar signature, and a representable window — the host runs the unit's `f0` and delivers
    /// the results back ([`CoopSched::deliver_jit_invoke_vals`]). `code` is the unit's code handle,
    /// `wasm` its emitted module, `argv` the raw i64 arg slots, `params`/`results` the unit entry's
    /// scalar signature, `mapped` the committed extent. A unit without emitted wasm (or a non-scalar
    /// signature / unrepresentable window) is serviced interpreted inside the pump and never surfaces.
    JitInvoke {
        code: i32,
        wasm: std::sync::Arc<[u8]>,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: u64,
    },
}

/// The cooperative multiplex scheduler's run-shared state, extracted from `drive` so that a future
/// resumable-across-FFI tier-up driver (#926 slice 2) can own it between host round-trips. `drive`
/// builds one with [`new`](CoopSched::new) and runs it to completion with [`pump`](CoopSched::pump);
/// the fields are exactly the run-shared locals `drive` used to hold — the task set, the run-shared
/// §12 fiber registry (+ its durable parallel arrays), the §14 confined child environments, the
/// fork/teardown bookkeeping, and the logical clock.
struct CoopSched {
    /// The live vCPUs: the root task (index 0) and its `thread.spawn`/`instantiate` descendants.
    tasks: Vec<TaskSlot>,
    /// §14 `instantiate` children's confined environments (handle = `env` index). The root and its
    /// `thread.spawn` siblings share `mem`/`host`/`dom.table` instead (`env == None`).
    extra_envs: Vec<ChildEnv>,
    /// The §12 fiber registry is **run-shared** (one handle namespace per domain) so a fiber created
    /// or suspended on one vCPU can be resumed on another (D57 migration).
    fibers: Vec<FiberState>,
    /// DURABILITY.md §12.8: each fiber's saved durable shadow-SP (run-shared, parallel to `fibers`;
    /// slot `s` is shadow context `s + 1`). Inert on a non-durable run.
    fiber_sp: Vec<u64>,
    /// Freeze residue (DURABILITY.md §12.8): each fiber's `(resolved entry func index, data-stack
    /// base)` — what a [`super::FrozenFiber`] needs after the fiber parks. Parallel to `fibers`.
    fiber_meta: Vec<(i32, i64)>,
    /// §12 teardown: child envs already torn down by a member's trap/exit (D37 death-is-revocation).
    dead_envs: std::collections::BTreeSet<usize>,
    /// FORK.md §9.2 — fork twins minted this run (task index = the pid a `clone_caller` returned).
    forked_twins: std::collections::BTreeSet<usize>,
    /// The scheduler's logical clock (advanced only when no task is runnable, to the earliest due
    /// `wait` deadline).
    clock: u64,
    /// #926 slice 2: wasm-JIT tier-up eligibility for this run's **module-0** tasks (the root and its
    /// same-module `thread.spawn` descendants). `None` ⇒ everything interprets, exactly the native
    /// `drive` (which never sets it). When set, each qualifying task's `Vm` carries it, so a direct
    /// module-0 `Call` to an eligible function surfaces as [`CoopStep::TierUp`] instead of interpreting
    /// the callee — the host runs the emitted `f{func}` and delivers the results back.
    eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 paged tier-up: the eligible set is page-checked (the emitted region carries a per-access
    /// page check), so an unrepresentable window surfaces with the reserved size instead of declining.
    page_checked: bool,
    /// The task currently paused on a surfaced tier-up, awaiting [`deliver_tierup`](Self::deliver_tierup):
    /// `(task index, caller-frame-relative dst slot, result types)`. At most one is ever outstanding —
    /// the driver services one tier-up round-trip before pumping again — so a single slot suffices.
    /// `None` between round-trips (and always, on the native driver).
    pending_tierup: Option<(usize, usize, Box<[ValType]>)>,
    /// The task currently paused on a surfaced §22 `Jit.invoke`, awaiting
    /// [`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals): `(task index, dst slot, result
    /// types)` — the same one-outstanding-round-trip discipline as `pending_tierup` (a tier-up and an
    /// invoke are never outstanding at once: each is one `pump` yield). `None` off the browser driver.
    pending_jit: Option<(usize, usize, Box<[ValType]>)>,
    /// #926 slice 2f — the driver-table **slot → code-handle mirror** for the browser B2 coop driver:
    /// `slot_codes[s]` is the §22 code handle a guest `Jit.install`ed at dispatch slot `s` (`-1`
    /// empty/natural), recorded at the pump's install/uninstall arms so the JS host can rebuild its
    /// `WebAssembly.Table` at each event boundary (the twin of the single-shot pump's `slot_codes`).
    /// Sized `1 << host.jit_table_log2()` — length 1 and unused on the native `drive` (no shared table).
    slot_codes: Vec<i32>,
    /// #1009: a generation counter bumped on each `Jit.install`/`Jit.uninstall` (the only `slot_codes`
    /// mutations) so the browser B2 driver rebuilds its `WebAssembly.Table` only when the mirror
    /// changed — a dispatch-heavy card that never installs syncs the table once, not per tier-up. The
    /// single-shot pump's `table_gen` twin (read via [`CoopRun::table_gen`]).
    table_gen: u32,
    /// #926 slice 2g — the **invoke-confined** fiber registry for a surfaced emitted `Jit.invoke`'s
    /// cross-tier bounces (the twin of [`Vcpu::invoke_fibers`]). While a `Jit.invoke` unit runs on the
    /// host, its `env.call_interp` callbacks share this registry across the invoke's several bounces (a
    /// fiber one callback parks is resumable by a later bounce of the *same* invoke), then it is cleared
    /// when the invoke resolves ([`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals) / `_trap`) —
    /// so an invoke's fibers die with it, exactly as the interpreted `run_invoke`'s loop-local registry
    /// does. A tier-up region's bounces use the run-level `fibers` instead (a parked fiber persists for
    /// the run to resume). Empty except during an outstanding invoke; always empty on the native `drive`.
    invoke_fibers: Vec<FiberState>,
}

impl CoopSched {
    /// Build the initial scheduler state: the root task at `entry`, plus any fibers a durable freeze
    /// left to re-seed (taken from `host.frozen_fibers`). This is `drive`'s former preamble verbatim
    /// — including the once-per-run entry-fuel charge — so its behaviour is unchanged. `eligible` /
    /// `page_checked` are the run's #926-slice-2 tier-up config: `None`/`false` (the native `drive`)
    /// leaves everything interpreting; a `Some` bitmap makes the root's — and its same-module
    /// `thread.spawn` descendants' — module-0 direct calls to eligible functions surface as tier-ups.
    fn new(
        dom: &Domain,
        entry: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut Host,
        tierup: Option<TierUpConfig>,
    ) -> Result<CoopSched, Trap> {
        // `page_checked` is meaningful only with a bitmap, so it rides in the same `Option`.
        let (eligible, page_checked) =
            tierup.map_or((None, false), |t| (Some(t.eligible), t.page_checked));
        // Fuel unification (safepoint-anchored): charge one fuel for *entering the top-level entry
        // function*, mirroring the per-callee-entry charge at `Op::Call`/`CallIndirect`/`TailCall*` and
        // the JIT's entry-prologue charge, so the tree-walker, bytecode, and JIT engines burn identically.
        // Gated exactly as the tree-walker's `drive` (`super::drive_arc`): a durable **thaw** re-enters to
        // continue an already-charged run (the root re-enters under `REWINDING`), so it must not re-charge.
        let is_thaw = host.is_durable()
            && mem
                .as_ref()
                .is_some_and(|m| m.durable_thaw_state(0) == super::STATE_REWINDING);
        if !is_thaw {
            *fuel = fuel.checked_sub(1).ok_or(Trap::OutOfFuel)?;
        }
        let mut tasks: Vec<TaskSlot> = vec![TaskSlot {
            vt: VTask::new(&dom.source.primary(), entry as usize, args)?,
            threads: Vec::new(),
            env: None,
            state: TaskState::Runnable,
        }];
        // #926 slice 2: arm the root task's `Vm` for tier-up. The entry runs in module 0, so a direct
        // call to an eligible function surfaces (`Vm::resume`'s `module == 0 && jit_eligible[callee]`
        // gate). `thread.spawn` children inherit the bitmap in `pump`'s `Spawn` arm (same-module only).
        if let Some(e) = &eligible {
            tasks[0].vt.active.jit_eligible = Some(std::sync::Arc::clone(e));
            tasks[0].vt.active.jit_page_checked = page_checked;
        }
        // §14 `instantiate` children's confined environments (handle = `env` index). The root and its
        // `thread.spawn` siblings use the shared `mem`/`host`/`dom.table` instead (`env == None`).
        let extra_envs: Vec<ChildEnv> = Vec::new();
        // The §12 fiber registry is **run-shared** (one handle namespace per domain) so a fiber created
        // or suspended on one vCPU can be resumed on another (D57 migration).
        let mut fibers: Vec<FiberState> = Vec::new();
        // DURABILITY.md §12.8: each fiber's saved durable shadow-SP (run-shared, parallel to `fibers`;
        // slot `s` is shadow context `s + 1`). Inert on a non-durable run.
        let mut fiber_sp: Vec<u64> = Vec::new();
        // Freeze residue (DURABILITY.md §12.8): each fiber's `(resolved entry func index, data-stack base)`
        // — what a [`super::FrozenFiber`] needs after the fiber parks (when its `Pending` `funcref`/`sp` are
        // gone). Parallel to `fibers`. Inert on a non-durable run.
        let mut fiber_meta: Vec<(i32, i64)> = Vec::new();
        // Thaw seeding (DURABILITY.md §12.8 slice 3.1.5): a `REWINDING` run re-creates the fibers a freeze
        // flattened *before* the root re-enters, so the root's re-issued `cont.resume` names the same dense
        // handles (0, 1, …) and each fiber's saved shadow-SP is back in `fiber_sp` for the swap to re-point
        // to. Taken (cleared) from the host; empty for a freeze or ordinary run.
        {
            let mut seed = std::mem::take(&mut host.frozen_fibers);
            seed.sort_by_key(|f| f.slot);
            for (expected, ff) in seed.into_iter().enumerate() {
                debug_assert_eq!(
                    expected,
                    fibers.len(),
                    "frozen fibers re-seed densely from slot 0"
                );
                debug_assert_eq!(
                    ff.slot,
                    fibers.len(),
                    "re-seeded slot matches the recorded handle"
                );
                fibers.push(FiberState::Pending {
                    funcref: ff.func,
                    sp: ff.sp,
                });
                fiber_sp.push(ff.shadow_sp);
                fiber_meta.push((ff.func, ff.sp));
            }
        }
        let clock: u64 = 0;
        // §12 "Domain lifetime & teardown" (owner 2026-07-24): child envs already torn down by a
        // member's trap/exit — a later live call through one completes with an errno instead of
        // parking forever (D37 death-is-revocation; the tree-walker's dead-callee park probe).
        let dead_envs: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        // FORK.md §9.2 — fork twins minted this run (task index = the pid a `clone_caller` returned). The
        // servicer-side `reap` (`wait`) acts only on ids in this allow-set (a foreign/bogus pid is
        // `-ECHILD`, never a park that hangs); an id is retired when reaped.
        let forked_twins: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

        Ok(CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            dead_envs,
            forked_twins,
            clock,
            eligible,
            page_checked,
            pending_tierup: None,
            pending_jit: None,
            // Sized to the domain table (`Domain::new(_, host.jit_table_log2())`), so a `Jit.install`'s
            // returned slot always indexes it. `1 << 0 == 1` and unused on the native `drive`.
            slot_codes: vec![-1i32; 1usize << host.jit_table_log2()],
            table_gen: 0,
            // Empty until a surfaced `Jit.invoke` bounces; populated only across that invoke's bounces.
            invoke_fibers: Vec::new(),
        })
    }

    /// Run the scheduler until it must pause — either the run finished ([`CoopStep::Done`]) or a task
    /// hit an eligible module-0 call and its emitted region must run on the host ([`CoopStep::TierUp`],
    /// resumed via [`deliver_tierup`](Self::deliver_tierup)). On the native `drive` (no eligibility),
    /// tier-up never fires, so this runs the whole schedule and returns `Done` — behaviourally the
    /// inline loop `drive` used to run. Each iteration services one runnable vCPU via `step_vcpu` and
    /// settles wakes/teardown; a run-fatal trap is `Err(trap)`.
    fn pump(
        &mut self,
        dom: &Domain,
        mem: &mut Option<Mem>,
        host: &mut Host,
        fuel: &mut u64,
        budget: u64,
    ) -> Result<CoopStep, Trap> {
        let CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            dead_envs,
            forked_twins,
            clock,
            eligible,
            page_checked,
            pending_tierup,
            pending_jit,
            slot_codes,
            table_gen,
            // The invoke-confined registry is threaded only by `CoopRun::bounce` (an emitted invoke's
            // callbacks), never touched by the scheduler loop itself.
            invoke_fibers: _,
        } = self;
        loop {
            // Domain lifetime & teardown (DESIGN.md §12 / ISSUES.md I37, owner 2026-07-24): a
            // member's trap/exit is terminal for its whole DOMAIN — run the teardown fixpoint
            // before reading the root's state, so a sibling's trap that killed the root domain
            // surfaces as the run's result (previously the root's timed wait simply outlived it).
            teardown_domains(tasks, extra_envs, dead_envs);
            // The root's result is the run's result (other vCPUs' effects are already reflected in it).
            if let TaskState::Done(res) = &tasks[0].state {
                let res = res.clone();
                // Freeze driver (DURABILITY.md §12.8 slice 3.1.4): a durable run left in `UNWINDING` has
                // drained the root's native stack into context 0's region; now flatten the still-parked
                // fibers into theirs, while the registry is alive, before the window is snapshotted. A drive
                // trap (out-of-scope fiber) surfaces as the run's result. `cont.*` durability is single-vCPU
                // (the entry guard refuses `thread.*`), so only the root task owns fibers.
                if res.is_ok()
                    && host.is_durable()
                    && mem.as_ref().map(|m| m.durable_state()) == Some(super::STATE_UNWINDING)
                {
                    let mut ctx = RunCtx {
                        table: &dom.table,
                        fuel: &mut *fuel,
                        mem: &mut *mem,
                        durable: true,
                        host: HostCell::Excl(&mut *host),
                    };
                    host.frozen_fibers =
                        freeze_drive(fibers, fiber_sp, fiber_meta, dom, &mut ctx, budget)?;
                }
                // `res` is the root's `Result<Vec<Value>, Trap>`: `Ok(vals)` → `Done(vals)`; a root
                // trap stays `Err(trap)` (the run's fatal trap), exactly as `drive` returned it.
                return res.map(CoopStep::Done);
            }
            // §3.6 (I36 slice 2) — settle wakes: a task parked on a live-call ticket wakes when the
            // callee's serve loop completed its dispatch; claiming the completion cell delivers the
            // reply (the tree-walker's cap_reply preference — a parked caller beats the cell).
            for t in tasks.iter_mut() {
                let hit = match &t.state {
                    TaskState::BlockedTicket {
                        ticket,
                        callee,
                        dst,
                    } => callee
                        .lock_unpoisoned()
                        .svc_results
                        .remove(ticket)
                        .map(|v| (v, *dst)),
                    _ => None,
                };
                if let Some((v, dst)) = hit {
                    t.vt.active.set(dst, Reg::from_i64(v));
                    t.state = TaskState::Runnable;
                }
            }
            // FORK.md §9.2 — reap wakes: a caller parked in `wait(pid)` wakes when fork twin `pid`
            // finishes, with the twin's exit status ([`super::reap_status`]; a trapped twin reaps as a
            // crash status, never a propagated trap — reap ≠ join). Two-phase (read the twin's outcome,
            // then deliver) so the caller and the twin task are not borrowed at once.
            let reap_wakes: Vec<(usize, u32, i64, usize)> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ci, t)| match &t.state {
                    TaskState::BlockedReap { pid, dst } => match &tasks[*pid].state {
                        TaskState::Done(res) => Some((ci, *dst, super::reap_status(res), *pid)),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            for (ci, dst, status, pid) in reap_wakes {
                tasks[ci].vt.active.set(dst, Reg::from_i64(status));
                tasks[ci].state = TaskState::Runnable;
                forked_twins.remove(&pid);
            }
            // I48 — wake blocking-resume idlers: a `TaskState::BlockedOnFiber { fiber }` becomes
            // runnable once its fiber is woken (the idle-timer's `WAIT_TIMED_OUT`, a `notify`'s
            // `WAIT_WOKEN`, or the cap-completion drain). Its cursor was rewound to the resume op, so
            // the next step re-executes it and claims the now-woken fiber. Centralized here so every
            // wake source feeds it uniformly (no per-site wiring). Runs before the pick so a fiber
            // woken during the previous step is seen this iteration.
            for t in tasks.iter_mut() {
                if let TaskState::BlockedOnFiber { fiber } = t.state {
                    if matches!(
                        fibers.get(fiber),
                        Some(FiberState::WaitParked { woken: Some(_), .. })
                            | Some(FiberState::CapParked { woken: Some(_), .. })
                    ) {
                        t.state = TaskState::Runnable;
                    }
                }
            }
            let Some(ti) = tasks
                .iter()
                .position(|t| matches!(t.state, TaskState::Runnable))
            else {
                // F2 (FIBER_PARK.md) — no runnable task with punt completions outstanding: that is
                // pending work on the offload pool, never a deadlock and never a reason to jump the
                // logical clock. Block on the store for the smallest outstanding id (submission
                // order), deliver through the ordered drain, and loop — the woken fibers'
                // resumers observe the wake at their next poll (or their own timers fire below on
                // a later pass).
                let min_cap = fibers
                    .iter()
                    .filter_map(|f| match f {
                        FiberState::CapParked {
                            id, woken: None, ..
                        } => Some(*id),
                        _ => None,
                    })
                    .min();
                if let Some(id) = min_cap {
                    let comps = host.completions();
                    let r = comps.wait(id);
                    for f in fibers.iter_mut() {
                        if let FiberState::CapParked {
                            id: fid,
                            woken: w @ None,
                            ..
                        } = f
                        {
                            if *fid == id {
                                *w = Some(r);
                            }
                        }
                    }
                    drain_cap_parked(fibers, &comps);
                    continue;
                }
                // No runnable task: fire the earliest `wait` timeout — whole-vCPU waiters and
                // event-parked fiber waiters alike (§3.6 slice 5a) — else it is a deadlock.
                let next = tasks
                    .iter()
                    .filter_map(|t| match t.state {
                        TaskState::BlockedWait { deadline, .. } => Some(deadline),
                        _ => None,
                    })
                    .chain(fibers.iter().filter_map(|f| match f {
                        FiberState::WaitParked {
                            deadline,
                            woken: None,
                            ..
                        } => Some(*deadline),
                        _ => None,
                    }))
                    .min();
                match next {
                    Some(d) => {
                        *clock = (*clock).max(d);
                        for t in tasks.iter_mut() {
                            if let TaskState::BlockedWait { deadline, dst, .. } = t.state {
                                if deadline <= *clock {
                                    t.vt.active.set(dst, Reg::from_i32(super::WAIT_TIMED_OUT));
                                    t.state = TaskState::Runnable;
                                }
                            }
                        }
                        // §3.6 slice 5a: a due fiber wait completes with `WAIT_TIMED_OUT` — the
                        // fiber becomes claimable (leaving the pending set, so this loop makes
                        // progress); its resumer's next `cont.resume` delivers the status.
                        for f in fibers.iter_mut() {
                            if let FiberState::WaitParked {
                                deadline,
                                woken: w @ None,
                                ..
                            } = f
                            {
                                if *deadline <= *clock {
                                    *w = Some(super::WAIT_TIMED_OUT);
                                }
                            }
                        }
                    }
                    None => return Err(Trap::ThreadFault), // deadlock (no runnable, no waiters)
                }
                continue;
            };

            // Select this vCPU's environment: the shared one (root + thread siblings), or its own
            // confined `instantiate` env. `tasks[ti].vt` and the chosen env borrow disjoint storage
            // (`tasks` vs `extra_envs` / the `mem`/`host`/`fuel` params), so the split borrow is sound.
            let mut ctx = match tasks[ti].env {
                None => RunCtx {
                    table: &dom.table,
                    fuel: &mut *fuel,
                    mem: &mut *mem,
                    durable: host.is_durable(),
                    host: HostCell::Excl(&mut *host),
                },
                Some(k) => {
                    let e = &mut extra_envs[k];
                    let durable = e
                        .host
                        .lock()
                        .unwrap_or_else(|er| er.into_inner())
                        .is_durable();
                    RunCtx {
                        table: &e.table,
                        fuel: &mut e.fuel,
                        mem: &mut e.mem,
                        durable,
                        host: HostCell::Shared(&e.host),
                    }
                }
            };
            let stop = step_vcpu(
                &mut tasks[ti].vt,
                fibers,
                fiber_sp,
                fiber_meta,
                dom,
                &mut ctx,
                budget,
                true, // the cooperative scheduler: idle blocking `cont.resume.block` (I48)
            );
            match stop {
                Err(trap) => complete(tasks, ti, Err(trap)),
                Ok(VcpuStop::Done(vals)) => complete(tasks, ti, Ok(vals)),
                // #926 slice 2 — wasm-JIT tier-up: this module-0 task hit a direct call to an eligible
                // function (its `Vm` carries the run's bitmap). `step_vcpu` has already spilled the frame
                // past the call, so the task is resumable with just the result slots filled. Stash the
                // delivery target — `(task, dst, result types)` — and surface the region to the driver;
                // `deliver_tierup` writes the emitted results back into `tasks[ti].vt.active` and the next
                // `pump` resumes the task. At most one tier-up is outstanding (we return here), so the
                // single `pending_tierup` slot suffices — no per-task field needed. On the native driver
                // no task is eligible, so this arm is never reached (the bitmap is `None`).
                Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                }) => {
                    debug_assert!(
                        pending_tierup.is_none(),
                        "a tier-up is already outstanding — deliver_tierup was skipped"
                    );
                    *pending_tierup = Some((ti, dst, results));
                    return Ok(CoopStep::TierUp { func, argv, mapped });
                }
                // Blocking stdin is only ever set on an owned-host `Vcpu` (the interactive session), never
                // a scheduler task — same rationale as tier-up above.
                Ok(VcpuStop::StdinPark) => {
                    unreachable!("blocking stdin not enabled on the scheduler driver")
                }
                // I48 — a blocking `cont.resume.block` of a still-parked fiber: idle this task on the
                // fiber (`step_vcpu` already rewound the resumer's cursor to the resume op). The
                // top-of-loop scan re-marks it `Runnable` once the fiber wakes.
                Ok(VcpuStop::BlockOnFiber { fiber }) => {
                    tasks[ti].state = TaskState::BlockedOnFiber { fiber };
                }
                // §3.6 (I36 slice 2) — the serve/call/offer trio, cooperative form.
                Ok(VcpuStop::SvcWait) => {
                    tasks[ti].state = TaskState::BlockedSvc;
                }
                Ok(VcpuStop::LiveCall {
                    ticket,
                    callee,
                    dst,
                }) => {
                    // The enqueue already happened in the op exec (holding only the callee's lock).
                    // Wake any svc.wait-parked task of the callee's domain — the tree-walker's
                    // `svc_wake` — then park the caller on its ticket.
                    let k = extra_envs
                        .iter()
                        .position(|e| std::sync::Arc::ptr_eq(&e.host, &callee));
                    // §12 teardown / D37 death-is-revocation (owner 2026-07-24): a call through an
                    // already-torn-down callee can never be replied — complete with the probeable
                    // errno instead of parking forever (the tree-walker's dead-callee park probe).
                    if k.is_some_and(|k| dead_envs.contains(&k)) {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i64(super::CAP_REVOKED));
                        continue;
                    }
                    if let Some(k) = k {
                        for t in tasks.iter_mut() {
                            if t.env == Some(k) && matches!(t.state, TaskState::BlockedSvc) {
                                t.state = TaskState::Runnable;
                            }
                        }
                    }
                    tasks[ti].state = TaskState::BlockedTicket {
                        ticket,
                        callee,
                        dst,
                    };
                }
                Ok(VcpuStop::CapPending { id, dst }) => {
                    // F2 (FIBER_PARK.md) — a punted dispatch, cooperative form. A FIBER parks
                    // (`CapParked` — the slice-5a contract; the ordered drain right after the
                    // park is the register-then-recheck closing the completion-raced-the-park
                    // window). The root keeps the inline wait (guest-invisible — the oracle
                    // parks the vCPU here instead; the cooperative driver has nothing else to
                    // run on this task anyway). Two more inline cases, both mirroring the
                    // oracle's predicate: a durable run (`freeze_drive` has no cap-park
                    // re-derivation — the freeze must never meet one) and a confined
                    // `instantiate` child (its completions live on ITS host; keeping the child
                    // inline keeps the drain single-store — recorded FIBER_PARK.md residue).
                    let durable = host.is_durable();
                    if tasks[ti].vt.active_id != ROOT_FIBER && !durable && tasks[ti].env.is_none() {
                        let comps = host.completions();
                        let k = tasks[ti].vt.active_id;
                        // I48: read the blocking-resume marker off the parking fiber's `Running` state
                        // before it is overwritten with `CapParked`.
                        let blocking_ip = match fibers.get(k) {
                            Some(FiberState::Running { blocking_ip }) => *blocking_ip,
                            _ => None,
                        };
                        let vt = &mut tasks[ti].vt;
                        let (rid, resumer, rdst) =
                            vt.chain.pop().expect("a running fiber has a resumer");
                        shadow_switch(mem, fiber_sp, &mut vt.root_shadow_sp, durable, k, rid);
                        let fvm = std::mem::replace(&mut vt.active, resumer);
                        fibers[k] = FiberState::CapParked {
                            vm: fvm,
                            dst,
                            id,
                            woken: None,
                        };
                        drain_cap_parked(fibers, &comps);
                        vt.active_id = rid;
                        // I48: a blocking resume idles the resumer on this cap-parked fiber. Rewind so
                        // the wake re-executes the resume; if the drain already claimed the completion,
                        // keep the task Runnable to re-resume at once, else park it until the drain
                        // (at idle or a later poll) wakes the fiber.
                        if let Some(ip) = blocking_ip {
                            vt.active.pc = ip;
                            let woken_now = matches!(
                                fibers.get(k),
                                Some(FiberState::CapParked { woken: Some(_), .. })
                            );
                            if !woken_now {
                                tasks[ti].state = TaskState::BlockedOnFiber { fiber: k };
                            }
                            continue;
                        }
                        vt.active.set(rdst, Reg::from_i32(super::FIBER_PARKED));
                        vt.active.set(rdst + 1, Reg::from_i64(0));
                    } else {
                        let comps = match tasks[ti].env {
                            None => host.completions(),
                            Some(k) => extra_envs[k].host.lock_unpoisoned().completions(),
                        };
                        let r = comps.wait(id);
                        tasks[ti].vt.active.set(dst, Reg::from_i64(r));
                    }
                }
                Ok(VcpuStop::ChildOffer { child, export, dst }) => {
                    // Mint a live-callee offer over a running child's export: shape from the
                    // CALLEE's module (fetched before the wirer's lock — the tree-walker's lock
                    // order), interned structurally into the wirer's table. A bad child handle /
                    // no such export is a probeable -EINVAL, matching the oracle.
                    let callee = usize::try_from(child)
                        .ok()
                        .and_then(|h| tasks[ti].threads.get(h).copied().flatten())
                        .and_then(|cidx| tasks[cidx].env)
                        .map(|k| std::sync::Arc::clone(&extra_envs[k].host));
                    let cap = callee.and_then(|callee: std::sync::Arc<std::sync::Mutex<Host>>| {
                        let sigs = callee.lock_unpoisoned().offer_shape(export)?;
                        match tasks[ti].env {
                            None => host.wire_live_impl(&callee, export, &sigs).ok(),
                            Some(pk) => extra_envs[pk]
                                .host
                                .lock_unpoisoned()
                                .wire_live_impl(&callee, export, &sigs)
                                .ok(),
                        }
                    });
                    tasks[ti]
                        .vt
                        .active
                        .set(dst, Reg::from_i32(cap.unwrap_or(super::EINVAL as i32)));
                }
                Ok(VcpuStop::CloneCaller {
                    reply_orig,
                    reply_twin,
                    dst,
                    has_result,
                }) => {
                    // FORK.md §9.2 — fork-returns-twice on the cooperative driver. Duplicate the caller
                    // parked on this handler's dispatch into a live **twin** (private window +
                    // duplicated powerbox, its own env), deliver `reply_twin` to the twin and
                    // `reply_orig` (pid mode: the twin's task id) to the original; both resume past the
                    // same fork `cap.call`. Fail-closed to a single reply on any shape the driver can't
                    // duplicate — never a hang, mirroring the oracle's degrade (svm-interp
                    // `fork_parked_caller`). `reap` (fork+wait) is a later slice — such modules fold.
                    let result: i64 = 'fork: {
                        // The running handler's dispatch ticket names the parked caller; outside a
                        // handler there is none → `-EINVAL`, exactly as the oracle.
                        let Some(ticket) = tasks[ti].vt.active.serve_ticket else {
                            break 'fork super::EINVAL;
                        };
                        // The server (this task) is the callee the caller parked on. In the fork
                        // topology the server is a spawned child with an `Arc` host (never the root).
                        let Some(server_env) = tasks[ti].env else {
                            break 'fork super::EINVAL;
                        };
                        let server_host = std::sync::Arc::clone(&extra_envs[server_env].host);
                        // Locate the parked caller on `(ticket, this server)`. In the cooperative driver
                        // the caller has already parked (it enqueued + woke us before we ran), so a miss
                        // is defensive → degrade.
                        let caller_ti = tasks.iter().position(|t| {
                            matches!(&t.state,
                            TaskState::BlockedTicket { ticket: tk, callee, .. }
                                if *tk == ticket && std::sync::Arc::ptr_eq(callee, &server_host))
                        });
                        let degrade = |tasks: &mut Vec<TaskSlot>,
                                       caller_ti: Option<usize>|
                         -> i64 {
                            // One reply to the caller, no twin. Explicit mode delivers `reply_orig`; pid
                            // mode delivers `-EAGAIN` (POSIX fork failure). Returns the handler's result.
                            let fallback = reply_orig.unwrap_or(super::EAGAIN);
                            if let Some(cti) = caller_ti {
                                if let TaskState::BlockedTicket { dst: cdst, .. } = tasks[cti].state
                                {
                                    tasks[cti].vt.active.set(cdst, Reg::from_i64(fallback));
                                    tasks[cti].state = TaskState::Runnable;
                                }
                            }
                            reply_orig.map_or(super::EAGAIN, |_| 0)
                        };
                        let Some(caller_ti) = caller_ti else {
                            break 'fork degrade(tasks, None);
                        };
                        let TaskState::BlockedTicket {
                            dst: caller_dst, ..
                        } = tasks[caller_ti].state
                        else {
                            break 'fork super::EINVAL;
                        };
                        let caller_env = tasks[caller_ti].env;
                        // Only a bare root caller (no spawned children/threads, no live fiber chain)
                        // forks faithfully — the oracle's `bare` gate. Anything else degrades.
                        let bare = tasks[caller_ti].threads.iter().all(|t| t.is_none())
                            && tasks[caller_ti].vt.active_id == ROOT_FIBER
                            && tasks[caller_ti].vt.chain.is_empty();
                        // Duplicate the caller's window (private copy — fork does not share memory) and
                        // powerbox (own handle namespace, shared `Arc` backings). A root caller (no env)
                        // or a non-forkable window/powerbox fails closed to a single reply.
                        // The twin's pid is its task index (`twin_ti` below); nothing is pushed between
                        // here and that push, so the fork factories learn it up front (#863 slice 2).
                        let twin_pid = tasks.len() as u64;
                        let forked = if bare {
                            caller_env.and_then(|ck| {
                                let twin_mem = match &extra_envs[ck].mem {
                                    Some(m) => Some(m.fork_private()?),
                                    None => None,
                                };
                                let twin_host = extra_envs[ck]
                                    .host
                                    .lock_unpoisoned()
                                    .fork_powerbox(twin_pid)?;
                                Some((ck, twin_mem, twin_host))
                            })
                        } else {
                            None
                        };
                        let Some((ck, twin_mem, twin_host)) = forked else {
                            break 'fork degrade(tasks, Some(caller_ti));
                        };
                        // The twin's continuation is the caller's — a bare root `Vm` cloned at its
                        // post-call resume point (`Vm` derives `Clone`; a bare caller carries no resume
                        // chain / invoke) — with `reply_twin` injected at the caller's reply slot.
                        let mut twin_active = tasks[caller_ti].vt.active.clone();
                        twin_active.set(caller_dst, Reg::from_i64(reply_twin));
                        let twin_vt = VTask {
                            active: twin_active,
                            active_id: ROOT_FIBER,
                            chain: Vec::new(),
                            root_shadow_sp: super::SHADOW_BASE,
                            active_invoke: None,
                            invoke_step_into: false,
                        };
                        // The twin is its own domain: a fresh env over the private window + duplicated
                        // powerbox, a natural table over the caller's (same-)module, the caller's env fuel.
                        let twin_table = build_table(dom.source.primary().progs.len(), 0);
                        let twin_eidx = extra_envs.len();
                        extra_envs.push(ChildEnv {
                            mem: twin_mem,
                            host: std::sync::Arc::new(std::sync::Mutex::new(twin_host)),
                            table: twin_table,
                            fuel: extra_envs[ck].fuel,
                        });
                        let twin_ti = tasks.len();
                        tasks.push(TaskSlot {
                            vt: twin_vt,
                            threads: Vec::new(),
                            env: Some(twin_eidx),
                            state: TaskState::Runnable,
                        });
                        // Mark the twin reapable so a later servicer-side `wait()` (`reap`) can deliver
                        // its exit status to the parent (FORK.md §8.6); retired when reaped.
                        forked_twins.insert(twin_ti);
                        // Deliver the original's reply and re-run it: explicit `reply_orig`, or pid mode
                        // = the twin's task id (parent-sees-pid). The handler's own return still writes
                        // the ticket's completion cell, but the caller is now `Runnable` (not
                        // `BlockedTicket`), so the settle scan never claims it — harmless, no flag needed.
                        let orig_reply = reply_orig.unwrap_or(twin_ti as i64);
                        tasks[caller_ti]
                            .vt
                            .active
                            .set(caller_dst, Reg::from_i64(orig_reply));
                        tasks[caller_ti].state = TaskState::Runnable;
                        twin_ti as i64
                    };
                    if has_result {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(result));
                    }
                }
                Ok(VcpuStop::Reap {
                    pid,
                    dst,
                    has_result,
                }) => {
                    // FORK.md §9.2 — the servicer side of `wait(pid)`. Reap fork twin `pid` on behalf of
                    // the caller parked on this handler's dispatch: deliver the twin's exit status now (if
                    // it finished) or park the caller until it does. `-ECHILD` for a pid this run did not
                    // mint (the handler's own reply carries it); `-EINVAL` outside a handler. Never a hang.
                    let result: i64 = 'reap: {
                        let Some(ticket) = tasks[ti].vt.active.serve_ticket else {
                            break 'reap super::EINVAL;
                        };
                        let Some(server_env) = tasks[ti].env else {
                            break 'reap super::EINVAL;
                        };
                        let server_host = std::sync::Arc::clone(&extra_envs[server_env].host);
                        let caller_ti = tasks.iter().position(|t| {
                            matches!(&t.state,
                            TaskState::BlockedTicket { ticket: tk, callee, .. }
                                if *tk == ticket && std::sync::Arc::ptr_eq(callee, &server_host))
                        });
                        // The pid must be a twin this run minted; otherwise a genuine `-ECHILD`, which the
                        // handler's own return delivers to the still-parked caller (the normal serve path).
                        let Some(pid_us) = usize::try_from(pid)
                            .ok()
                            .filter(|p| forked_twins.contains(p))
                        else {
                            break 'reap super::ECHILD;
                        };
                        // The cooperative driver parks the caller before the handler runs, so a miss is
                        // defensive → `-EAGAIN` (retryable, never a false `-ECHILD` — the twin is real).
                        let Some(caller_ti) = caller_ti else {
                            break 'reap super::EAGAIN;
                        };
                        let TaskState::BlockedTicket {
                            dst: caller_dst, ..
                        } = tasks[caller_ti].state
                        else {
                            break 'reap super::EINVAL;
                        };
                        // Twin finished → deliver its status now and retire it; else park the caller on it
                        // (the settle scan wakes it on twin-exit). Either way the caller's reply is handled
                        // here — the handler's own return lands on a caller no longer `BlockedTicket`, so
                        // the settle scan never claims it (harmless, mirroring `clone_caller`).
                        if let TaskState::Done(res) = &tasks[pid_us].state {
                            let status = super::reap_status(res);
                            forked_twins.remove(&pid_us);
                            tasks[caller_ti]
                                .vt
                                .active
                                .set(caller_dst, Reg::from_i64(status));
                            tasks[caller_ti].state = TaskState::Runnable;
                            status
                        } else {
                            tasks[caller_ti].state = TaskState::BlockedReap {
                                pid: pid_us,
                                dst: caller_dst,
                            };
                            0
                        }
                    };
                    if has_result {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(result));
                    }
                }
                Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module,
                }) => {
                    // `func` resolves in the SPAWNING FRAME's module (an installed §22 unit spawns its
                    // own functions — CONSOLIDATION.md §11); the child's root frame starts there too.
                    let Some(cm) = dom.source.get(module as usize) else {
                        complete(tasks, ti, Err(Trap::Malformed));
                        continue;
                    };
                    if func as usize >= cm.progs.len() {
                        complete(tasks, ti, Err(Trap::Malformed));
                        continue;
                    }
                    let live = tasks
                        .iter()
                        .filter(|t| !matches!(t.state, TaskState::Done(_)))
                        .count();
                    if live >= super::MAX_VCPUS {
                        complete(tasks, ti, Err(Trap::ThreadFault)); // thread bomb
                        continue;
                    }
                    let mut child =
                        VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
                    child.active.module = module as usize;
                    child.active.home = module as usize;
                    // #926 slice 2: a `thread.spawn` child running in **module 0** tiers up its own
                    // eligible calls too (the run's bitmap is per-module-0-function, shared across the
                    // root and its same-module threads). A child spawned in another module (a confined
                    // §22 unit spawning its own function) runs a different module, where the module-0
                    // bitmap does not apply — leave it interpreting.
                    if module == 0 {
                        if let Some(e) = eligible.as_ref() {
                            child.active.jit_eligible = Some(std::sync::Arc::clone(e));
                            child.active.jit_page_checked = *page_checked;
                        }
                    }
                    let cidx = tasks.len();
                    // §12 seed the child vCPU's TLS register to its dense id (root is task 0), so
                    // `vcpu.tls.get` returns the worker index — the tree-walker's `tls: id` seeding.
                    child.active.tls = cidx as i64;
                    // A thread shares its spawner's window/powerbox — so it inherits the spawner's env
                    // (the shared domain for a root-spawned thread, or the same confined `instantiate`
                    // env for one spawned by a confined child).
                    let env = tasks[ti].env;
                    tasks.push(TaskSlot {
                        vt: child,
                        threads: Vec::new(),
                        env,
                        state: TaskState::Runnable,
                    });
                    let handle = tasks[ti].threads.len() as i32;
                    tasks[ti].threads.push(Some(cidx));
                    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
                }
                Ok(VcpuStop::Instantiate {
                    ibase,
                    isize: isz,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                }) => {
                    // Validate the child entry signature against module 0 (a same-module child): it
                    // returns one `i64` and takes either its `Instantiator` (one `i64`) or its
                    // `Instantiator`+`AddressSpace` (two) — its starter caps over its own window.
                    let c0 = dom.source.primary();
                    let want_as = c0
                        .sigs
                        .get(entry as usize)
                        .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                    let ok_entry = c0
                        .sigs
                        .get(entry as usize)
                        .is_some_and(|(p, r)| child_entry_ok(p, r));
                    // The carve must be a power-of-two-aligned sub-window within `[0, isize)` — a child
                    // gets only what the holder sub-allocates (§14/D19).
                    let child_size = if (0..64).contains(&size_log2) {
                        1u64 << size_log2
                    } else {
                        0
                    };
                    let off_u = off as u64;
                    let fits = carve_fits(
                        off_u,
                        size_log2,
                        isz,
                        ibase,
                        mem.as_ref().map_or(0, |m| m.null_guard),
                    );
                    if !ok_entry || !fits {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                    let live = tasks
                        .iter()
                        .filter(|t| !matches!(t.state, TaskState::Done(_)))
                        .count();
                    if live >= super::MAX_VCPUS {
                        complete(tasks, ti, Err(Trap::ThreadFault)); // instantiate bomb
                        continue;
                    }
                    // The parent's window base (holder-relative `ibase`/`off` → backing-absolute, so
                    // nesting composes) and fuel (the child's quota is sub-allocated from, and capped by,
                    // the parent's) come from the parent's environment.
                    let (pbase, pfuel) = match tasks[ti].env {
                        None => (mem.as_ref().map_or(0, |m| m.window.base()), *fuel),
                        Some(k) => (
                            extra_envs[k].mem.as_ref().map_or(0, |m| m.window.base()),
                            extra_envs[k].fuel,
                        ),
                    };
                    let abs_base = pbase + ibase + off_u;
                    let child_mem = match tasks[ti].env {
                        None => mem
                            .as_ref()
                            .map(|m| m.nested_view(abs_base, size_log2 as u8)),
                        Some(k) => extra_envs[k]
                            .mem
                            .as_ref()
                            .map(|m| m.nested_view(abs_base, size_log2 as u8)),
                    };
                    // Attenuated powerbox: an `Instantiator` (so the child can itself nest — confinement
                    // composes to any depth) and an `AddressSpace` (so it manages its own pages), each
                    // over its *own* `[0, child_size)` window — its entry arguments. op 0 grants only
                    // those two; op 11 (`grants` is `Some((ptr, n))`) additionally re-grants a by-name cap
                    // list read from the parent window, so a spawned stage resolves an inherited region
                    // (a ring end) by name — the concurrent-pipeline spawn. The named build fails closed
                    // via the shared, fuzzed `spawn_named_child` (mirrors the op-13 arm; grants resolve
                    // against the root `host`, so a confined child's forged handle fails `can_regrant`).
                    let (mut child_host, cinst, cas) = if let Some((grants_ptr, grants_n)) = grants
                    {
                        // Parse `grants_n × 16-byte {name_off:u32, name_len:u32, handle:i32, flags:u32}`
                        // records from the parent window (identical to the op-13 `InstantiateModule` arm).
                        let pm: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        let list: Result<Vec<(String, i32)>, Trap> = (|| {
                            let m = pm.ok_or(Trap::Malformed)?;
                            let mut list: Vec<(String, i32)> = Vec::new();
                            for i in 0..grants_n {
                                let rec = m.read_window(grants_ptr + i * 16, 16)?;
                                let name_off =
                                    u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
                                let name_len =
                                    u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                                let handle = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                                let name_bytes = m.read_window(name_off, name_len)?;
                                let name =
                                    String::from_utf8(name_bytes).map_err(|_| Trap::CapFault)?;
                                list.push((name, handle));
                            }
                            Ok(list)
                        })();
                        let list = match list {
                            Ok(l) => l,
                            Err(t) => {
                                complete(tasks, ti, Err(t));
                                continue;
                            }
                        };
                        match host.spawn_named_child(&list, child_size) {
                            Some(triple) => triple,
                            None => {
                                complete(tasks, ti, Err(Trap::CapFault));
                                continue;
                            }
                        }
                    } else {
                        let mut ch = Host::new();
                        let cinst = ch.grant_instantiator(0, child_size);
                        let cas = ch.grant_address_space(0, child_size);
                        (ch, cinst, cas)
                    };
                    // §3.6: a same-module child serves over the shared program — its serve machinery
                    // (enqueue admission, handler resolution) and any `child_offer` shape read the
                    // domain's registered module, exactly the tree-walker's `self_module` handoff.
                    child_host.self_module = match tasks[ti].env {
                        None => host.self_module.clone(),
                        Some(k) => extra_envs[k].host.lock_unpoisoned().self_module.clone(),
                    };
                    let child_args = if want_as {
                        vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                    } else {
                        vec![Value::I64(cinst as i64)]
                    };
                    // §3d: a record's budget funds the child here — the commit site, after every
                    // other refusal (geometry, grants), so a refused spawn leaves it intact.
                    let child_fuel = if budget != 0 {
                        match take_spawn_budget(host, budget, child_size, pfuel) {
                            Err(t) => {
                                complete(tasks, ti, Err(t));
                                continue;
                            }
                            Ok(None) => {
                                tasks[ti]
                                    .vt
                                    .active
                                    .set(dst, Reg::from_i32(super::EINVAL as i32));
                                continue;
                            }
                            Ok(Some(f)) => f,
                        }
                    } else if quota <= 0 {
                        pfuel
                    } else {
                        (quota as u64).min(pfuel)
                    };
                    // A nested child is its **own** domain: a fresh natural table over module 0 (no access
                    // to installed §22 units — matching the tree-walker's `DomainTable::new(&cfuncs, 0)`).
                    let c0 = dom.source.primary();
                    let child_table = build_table(c0.progs.len(), 0);
                    let child_vt = VTask::new(&c0, entry as usize, &child_args)?;
                    let eidx = extra_envs.len();
                    extra_envs.push(ChildEnv {
                        mem: child_mem,
                        host: std::sync::Arc::new(std::sync::Mutex::new(child_host)),
                        table: child_table,
                        fuel: child_fuel,
                    });
                    let cidx = tasks.len();
                    tasks.push(TaskSlot {
                        vt: child_vt,
                        threads: Vec::new(),
                        env: Some(eidx),
                        state: TaskState::Runnable,
                    });
                    let handle = tasks[ti].threads.len() as i32;
                    tasks[ti].threads.push(Some(cidx));
                    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
                }
                Ok(VcpuStop::InstantiateModule {
                    ibase,
                    isize: isz,
                    mh,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                    budget,
                }) => {
                    // Resolve the granted Module (a forged/closed/wrong-type handle is an inert CapFault).
                    let (cfuncs, cmem_log2, cdata, cmodule) = match host.resolve_module(mh) {
                        Ok(g) => (
                            g.funcs.clone(),
                            g.memory_log2,
                            g.data.clone(),
                            std::sync::Arc::clone(&g.module),
                        ),
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    // Compile the granted module to bytecode. A module using an op the engine can't lower
                    // is the one place a guest-provided program outruns coverage (no tree-walker fallback
                    // mid-run) — a `Malformed` trap, exactly as for `Jit.install`.
                    let child_compiled = match compile_module(&cfuncs) {
                        Some(c) => c,
                        None => {
                            complete(tasks, ti, Err(Trap::Malformed));
                            continue;
                        }
                    };
                    // The child entry sig is validated against the *child module*. A separate-module
                    // child's carve must equal its declared memory (§14 transparency: it runs exactly as
                    // it would standalone — same window size, same wrap behaviour).
                    let want_as = child_compiled
                        .sigs
                        .get(entry as usize)
                        .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                    let ok_entry = child_compiled
                        .sigs
                        .get(entry as usize)
                        .is_some_and(|(p, r)| child_entry_ok(p, r));
                    let child_size = if (0..64).contains(&size_log2) {
                        1u64 << size_log2
                    } else {
                        0
                    };
                    let off_u = off as u64;
                    let fits = carve_fits(
                        off_u,
                        size_log2,
                        isz,
                        ibase,
                        mem.as_ref().map_or(0, |m| m.null_guard),
                    );
                    let mod_ok = cmem_log2 == Some(size_log2 as u8);
                    if !ok_entry || !fits || !mod_ok {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                    let live = tasks
                        .iter()
                        .filter(|t| !matches!(t.state, TaskState::Done(_)))
                        .count();
                    if live >= super::MAX_VCPUS {
                        complete(tasks, ti, Err(Trap::ThreadFault));
                        continue;
                    }
                    let (pbase, pfuel) = match tasks[ti].env {
                        None => (mem.as_ref().map_or(0, |m| m.window.base()), *fuel),
                        Some(k) => (
                            extra_envs[k].mem.as_ref().map_or(0, |m| m.window.base()),
                            extra_envs[k].fuel,
                        ),
                    };
                    let abs_base = pbase + ibase + off_u;
                    // Build the child window and materialize the module's data segments into the carve
                    // (exactly as if the child wrote them; the verifier bounded them to its declared window
                    // == the carve). RO protection of `readonly` segments is skipped for nested children
                    // (intra-domain self-corruption is a §1 non-goal), matching the tree-walker.
                    let child_mem = {
                        let pm: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        if let Some(m) = pm {
                            for d in cdata.iter() {
                                if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
                                    for (k, &b) in d.bytes.iter().enumerate() {
                                        m.set_byte(abs_base + d.offset + k as u64, b);
                                    }
                                }
                            }
                        }
                        pm.map(|m| m.nested_view(abs_base, size_log2 as u8))
                    };
                    // op 5 grants only Instantiator+AddressSpace; op 13 (`grants` is `Some((ptr, n))`)
                    // additionally re-grants a by-name cap list read from the parent window, so a spawned
                    // command resolves an inherited `stdout` by name (STAGE1.md — the shell "exec"
                    // primitive). The named build fails closed via the shared, fuzzed `spawn_named_child`.
                    let (mut child_host, cinst, cas) = if let Some((grants_ptr, grants_n)) = grants
                    {
                        // Parse `grants_n × 16-byte {name_off:u32, name_len:u32, handle:i32, flags:u32}`
                        // records from the parent window (mirrors the tree-walk op-13 arm in lib.rs).
                        let pm: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        let list: Result<Vec<(String, i32)>, Trap> = (|| {
                            let m = pm.ok_or(Trap::Malformed)?;
                            let mut list: Vec<(String, i32)> = Vec::new();
                            for i in 0..grants_n {
                                let rec = m.read_window(grants_ptr + i * 16, 16)?;
                                let name_off =
                                    u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
                                let name_len =
                                    u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                                let handle = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                                let name_bytes = m.read_window(name_off, name_len)?;
                                let name =
                                    String::from_utf8(name_bytes).map_err(|_| Trap::CapFault)?;
                                list.push((name, handle));
                            }
                            Ok(list)
                        })();
                        let list = match list {
                            Ok(l) => l,
                            Err(t) => {
                                complete(tasks, ti, Err(t));
                                continue;
                            }
                        };
                        match host.spawn_named_child(&list, child_size) {
                            Some(triple) => triple,
                            None => {
                                complete(tasks, ti, Err(Trap::CapFault));
                                continue;
                            }
                        }
                    } else {
                        let mut ch = Host::new();
                        let cinst = ch.grant_instantiator(0, child_size);
                        let cas = ch.grant_address_space(0, child_size);
                        (ch, cinst, cas)
                    };
                    // §3.6: a separate-module child serves its OWN offers — enqueue admission,
                    // handler resolution, and `child_offer` shape all read its module (tree-walk
                    // lockstep: the spawn sets `self_module` from the grant).
                    child_host.set_self_module(&cmodule);
                    // IMPORTS.md phase 3 / §3.3: bind the child module's import manifest against its
                    // granted powerbox — a chibicc child's generic imports (`write`/`read`/`exit`, and any
                    // named grant) resolve here, so a compiled command actually does I/O rather than
                    // `CapFault`ing on its first `write`. `spawn_named_child` registers the *names* but does
                    // not bind the manifest, so the driver does it (the same `bind_child_manifest` the tree-
                    // walker's op-13 arm and the JIT's `child_bind_imports` hook call). A `required` slot
                    // with nothing to bind fails the spawn closed with a probeable `-EINVAL` (as the tree-
                    // walker does), never a trap. Empty for a manifest-free child (imports is empty → Ok).
                    if child_host
                        .bind_child_manifest(&cmodule.imports, &cmodule.types)
                        .is_err()
                    {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                    let child_args = if want_as {
                        vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                    } else {
                        vec![Value::I64(cinst as i64)]
                    };
                    // §3d: a record's budget funds the child here — the commit site, after every
                    // other refusal (module resolve, geometry, grants, manifest binding).
                    let child_fuel = if budget != 0 {
                        match take_spawn_budget(host, budget, child_size, pfuel) {
                            Err(t) => {
                                complete(tasks, ti, Err(t));
                                continue;
                            }
                            Ok(None) => {
                                tasks[ti]
                                    .vt
                                    .active
                                    .set(dst, Reg::from_i32(super::EINVAL as i32));
                                continue;
                            }
                            Ok(Some(f)) => f,
                        }
                    } else if quota <= 0 {
                        pfuel
                    } else {
                        (quota as u64).min(pfuel)
                    };
                    // Push the child's compiled module and run the child over it — its own domain: a
                    // natural table mapping into *its* module index (no installed §22 units).
                    let progs_len = child_compiled.progs.len();
                    let cm = dom.source.push(child_compiled);
                    let child_table = build_table_for(progs_len, 0, cm as u32);
                    let cunit = dom.source.get(cm).ok_or(Trap::Malformed)?;
                    let mut child_vt = VTask::new(&cunit, entry as usize, &child_args)?;
                    child_vt.active.module = cm;
                    child_vt.active.home = cm;
                    let eidx = extra_envs.len();
                    extra_envs.push(ChildEnv {
                        mem: child_mem,
                        host: std::sync::Arc::new(std::sync::Mutex::new(child_host)),
                        table: child_table,
                        fuel: child_fuel,
                    });
                    let cidx = tasks.len();
                    tasks.push(TaskSlot {
                        vt: child_vt,
                        threads: Vec::new(),
                        env: Some(eidx),
                        state: TaskState::Runnable,
                    });
                    let handle = tasks[ti].threads.len() as i32;
                    tasks[ti].threads.push(Some(cidx));
                    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
                }
                Ok(VcpuStop::Join { handle, dst }) => {
                    let slot = match super::resolve_thread(&tasks[ti].threads, handle) {
                        Ok(s) => s,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let child = tasks[ti].threads[slot].expect("resolve_thread checked liveness");
                    match &tasks[child].state {
                        TaskState::Done(res) => {
                            // The child already finished: deliver now (a child trap propagates here).
                            let res = res.clone();
                            tasks[ti].threads[slot] = None;
                            match res {
                                Ok(vals) => {
                                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                                    tasks[ti].vt.active.set(dst, Reg::from_value(v));
                                }
                                Err(t) => complete(tasks, ti, Err(t)),
                            }
                        }
                        _ => {
                            tasks[ti].state = TaskState::BlockedJoin { child, slot, dst };
                        }
                    }
                }
                Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                }) => {
                    // §3.6 slice 5a: a wait issued INSIDE a fiber parks the FIBER, not this vCPU
                    // (the tree-walk oracle's fiber-park routing — DESIGN.md "blocks the fiber,
                    // never the domain"; `fiber_parks.rs`). Unwind one chain link to the resumer
                    // with `(FIBER_PARKED, 0)` and set the fiber aside; the park-time value recheck
                    // closes the park-vs-store race (a store that already landed wakes it with
                    // `WAIT_NOT_EQUAL` — after the one transient `FIBER_PARKED`, like the oracle).
                    if tasks[ti].vt.active_id != ROOT_FIBER {
                        let durable = host.is_durable();
                        let k = tasks[ti].vt.active_id;
                        // I48: read the blocking-resume marker off the parking fiber's `Running` state
                        // (set at the claim) before it is overwritten with `WaitParked` below.
                        let blocking_ip = match fibers.get(k) {
                            Some(FiberState::Running { blocking_ip }) => *blocking_ip,
                            _ => None,
                        };
                        let vt = &mut tasks[ti].vt;
                        let (rid, resumer, rdst) =
                            vt.chain.pop().expect("a running fiber has a resumer");
                        shadow_switch(mem, fiber_sp, &mut vt.root_shadow_sp, durable, k, rid);
                        let fvm = std::mem::replace(&mut vt.active, resumer);
                        let cur = mem
                            .as_ref()
                            .map(|m| m.atomic_value(base, width))
                            .unwrap_or(0);
                        let woken = (cur != expected).then_some(super::WAIT_NOT_EQUAL);
                        fibers[k] = FiberState::WaitParked {
                            vm: fvm,
                            wait_dst: dst,
                            key: base,
                            deadline: clock.saturating_add(timeout),
                            real_deadline: sched_wall_deadline(timeout),
                            woken,
                        };
                        vt.active_id = rid;
                        // I48: a blocking resume idles the resumer on this fiber instead of the
                        // FIBER_PARKED poll. Rewind so the wake re-executes the resume; if the
                        // value-recheck already woke the fiber, keep the task Runnable to re-resume at
                        // once (the oracle's recheck-re-admit — no transient poll), else park it until
                        // the fiber's event/idle-timer wakes it.
                        if let Some(ip) = blocking_ip {
                            vt.active.pc = ip;
                            if woken.is_none() {
                                tasks[ti].state = TaskState::BlockedOnFiber { fiber: k };
                            }
                            continue;
                        }
                        vt.active.set(rdst, Reg::from_i32(super::FIBER_PARKED));
                        vt.active.set(rdst + 1, Reg::from_i64(0));
                        continue;
                    }
                    // Re-read the value (the cooperative analogue of the futex compare-under-lock): if it
                    // already changed, return not-equal; else park until notified or timed out. Both the
                    // value re-read and the rendezvous key are taken against THIS task's own memory: a
                    // confined `instantiate` child steps against its `extra_envs` window, not the root
                    // `mem`, and the key is backing-identity canonical (`futex_key`) so two children that
                    // mapped the same `SharedRegion` into separate windows rendezvous (S1c). Reading the
                    // root `mem` here instead would make a child's `wait` on its mapped ring flag re-read
                    // an unrelated root byte and spin forever.
                    let (cur, key) = {
                        let tmem: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        (
                            tmem.map(|m| m.atomic_value(base, width)).unwrap_or(0),
                            tmem.map(|m| m.futex_key(base))
                                .unwrap_or(super::FutexKey::Anon(base)),
                        )
                    };
                    if cur != expected {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::WAIT_NOT_EQUAL));
                    } else {
                        tasks[ti].state = TaskState::BlockedWait {
                            key,
                            deadline: clock.saturating_add(timeout),
                            dst,
                        };
                    }
                }
                Ok(VcpuStop::Notify { base, count, dst }) => {
                    // Wake up to `count` waiters, lowest task index first (deterministic). Key on the
                    // notifying task's own memory + backing identity (mirrors the wait arm), so a notify
                    // from one child's window matches a waiter parked from another child's window on the
                    // same `SharedRegion` byte.
                    let key = {
                        let tmem: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        tmem.map(|m| m.futex_key(base))
                            .unwrap_or(super::FutexKey::Anon(base))
                    };
                    let want = count as u32;
                    let mut woken = 0u32;
                    for t in tasks.iter_mut() {
                        if woken >= want {
                            break;
                        }
                        if let TaskState::BlockedWait {
                            key: wkey,
                            dst: wdst,
                            ..
                        } = t.state
                        {
                            if wkey == key {
                                t.vt.active.set(wdst, Reg::from_i32(super::WAIT_WOKEN));
                                t.state = TaskState::Runnable;
                                woken += 1;
                            }
                        }
                    }
                    // §3.6 slice 5a: also wake event-parked FIBER waiters (lowest slot next, deterministic
                    // like the task scan). Fibers can't coexist with `instantiate` (the module-level veto),
                    // so a fibered wait is always single-window — it keys on the raw confined address,
                    // unchanged. The status is delivered when a `cont.resume` claims the fiber.
                    for f in fibers.iter_mut() {
                        if woken >= want {
                            break;
                        }
                        if let FiberState::WaitParked {
                            key: fkey,
                            woken: w @ None,
                            ..
                        } = f
                        {
                            if *fkey == base {
                                *w = Some(super::WAIT_WOKEN);
                                woken += 1;
                            }
                        }
                    }
                    tasks[ti].vt.active.set(dst, Reg::from_i32(woken as i32));
                }
                Ok(VcpuStop::JitInstall { h, code, dst }) => {
                    // Resolve authority + the unit's funcs from the host (a forged/cross-domain handle is
                    // an inert CapFault → trap), compile the unit to bytecode, and install it. Compiling
                    // the unit can fail only if it uses an op the bytecode engine doesn't lower yet — the
                    // one place a guest-provided unit can outrun coverage (no tree-walker fallback mid-run).
                    let funcs = match host.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = host.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
                    }) {
                        Ok(f) => f,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let res = match compile_module(&funcs) {
                        Some(unit) => match dom.install(unit) {
                            Some(slot) => {
                                // #926 slice 2f: mirror `slot → code` so the browser B2 driver can
                                // rebuild its `WebAssembly.Table` at the next event boundary (installs
                                // only ever happen between host events — a unit with a `cap.call` never
                                // emits, so the install itself always runs interpreted). Twin of the
                                // single-shot pump's `slot_codes` recording; inert on the native drive.
                                if let Some(e) = slot_codes.get_mut(slot) {
                                    *e = code;
                                }
                                *table_gen = table_gen.wrapping_add(1); // slot mirror changed → re-sync
                                slot as i64
                            }
                            None => super::ENOSPC,
                        },
                        None => {
                            complete(tasks, ti, Err(Trap::Malformed)); // unit op outside coverage
                            continue;
                        }
                    };
                    tasks[ti].vt.active.set(dst, Reg::from_i64(res));
                }
                Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                    if let Err(t) = host.resolve_jit_domain(h) {
                        complete(tasks, ti, Err(t)); // authority check
                        continue;
                    }
                    let n_real = dom.source.primary().progs.len();
                    let res = if dom.uninstall(slot as usize, n_real) {
                        // Keep the B2 mirror exact — a freed slot must trap in the JS table too.
                        if let Some(e) = slot_codes.get_mut(slot as usize) {
                            *e = -1;
                        }
                        *table_gen = table_gen.wrapping_add(1); // slot mirror changed → re-sync
                        0
                    } else {
                        super::EINVAL
                    };
                    tasks[ti].vt.active.set(dst, Reg::from_i64(res));
                }
                Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                }) => {
                    // #926 slice 2e — surface to the browser host when this run drives tier-up
                    // (`eligible` set) and the unit has **emitted wasm** with an all-scalar signature
                    // over a representable window: the host runs the emitted `f0` and delivers the
                    // results back ([`deliver_jit_invoke_vals`]). Otherwise — the native `drive` (no
                    // `eligible`), an interpreter-only unit (no emitted wasm), a non-scalar signature,
                    // or an unrepresentable window — fall through to the interpreted service below,
                    // which is always correct (the fail-closed default the single-vCPU pump also uses).
                    let scalar = |t: &ValType| {
                        matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64)
                    };
                    let emittable = eligible.is_some()
                        && params.iter().all(scalar)
                        && results.iter().all(scalar);
                    let surfaced = if emittable {
                        // Resolve the unit's emitted wasm exactly as the browser FFI's resolver does
                        // (`jit_unit_wasm`), and the #717 committed-extent bound over the run's window
                        // (a `Jit.invoke` runs against the shared root powerbox/window).
                        let wasm = host.resolve_jit_domain(h).ok().and_then(|domain| {
                            let (cd, cu) = host.resolve_jit_code(code).ok()?;
                            (cd == domain).then(|| host.jit_unit_wasm(cd, cu))?
                        });
                        let mapped = match mem.as_ref() {
                            None => Some(0),
                            Some(m) if *page_checked => Some(m.reserved_size()),
                            Some(m) => m.scalar_extent(),
                        };
                        wasm.zip(mapped)
                    } else {
                        None
                    };
                    if let Some((wasm, mapped)) = surfaced {
                        *pending_jit = Some((ti, dst as usize, results.clone()));
                        return Ok(CoopStep::JitInvoke {
                            code,
                            wasm,
                            argv,
                            params,
                            results,
                            mapped,
                        });
                    }
                    // Resolve unit funcs (authority + cross-domain) and compile, as for install.
                    let funcs = match host.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = host.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
                    }) {
                        Ok(f) => f,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let unit = match compile_module(&funcs) {
                        Some(u) => u,
                        None => {
                            complete(tasks, ti, Err(Trap::Malformed));
                            continue;
                        }
                    };
                    // Arity-check the unit entry (func 0) against the call's (code-stripped) signature.
                    let arity_ok = unit.sigs.first().is_some_and(|(ep, er)| {
                        ep.len() == params.len() && er.len() == results.len()
                    });
                    if !arity_ok {
                        complete(tasks, ti, Err(Trap::CapFault));
                        continue;
                    }
                    // Marshal args via the slot ABI, push the unit as a transient module, run it.
                    let child_args: Vec<Value> = params
                        .iter()
                        .zip(argv.iter())
                        .map(|(ty, s)| slot_to_val(*ty, *s))
                        .collect();
                    let umod = dom.source.push(unit);
                    match run_invoke(
                        &dom.source,
                        &dom.table,
                        umod,
                        &child_args,
                        fuel,
                        mem,
                        &mut HostCell::Excl(host),
                    ) {
                        Ok(vals) => {
                            for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                                let re = slot_to_val(*ty, val_to_slot(*v));
                                tasks[ti].vt.active.set(dst + i as u32, Reg::from_value(re));
                            }
                        }
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// #926 slice 2 — deliver an emitted tier-up region's results into the paused task and clear the
    /// pending slot, so the next [`pump`](Self::pump) resumes it. `vals` are the raw i64 result slots
    /// the host read out of the emitted `f{func}` run; they are re-tagged into the caller frame's `dst`
    /// slots per the recorded result types (mirroring [`Vcpu::deliver_tierup`]). A short reply is a
    /// malformed host reply, which traps the task (its domain tears down, surfacing as the run result).
    fn deliver_tierup(&mut self, vals: &[i64]) {
        let (ti, dst, results) = self
            .pending_tierup
            .take()
            .expect("deliver_tierup with no pending tier-up");
        if vals.len() < results.len() {
            complete(&mut self.tasks, ti, Err(Trap::Malformed));
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.tasks[ti].vt.active.set(
                dst as u32 + i as u32,
                Reg::from_value(slot_to_val(*ty, vals[i])),
            );
        }
    }

    /// #926 slice 2 — the emitted tier-up region trapped: surface it exactly where the interpreter
    /// would by trapping the paused task (mirroring [`Vcpu::deliver_tierup_trap`]).
    fn deliver_tierup_trap(&mut self, trap: Trap) {
        let (ti, _dst, _results) = self
            .pending_tierup
            .take()
            .expect("deliver_tierup_trap with no pending tier-up");
        complete(&mut self.tasks, ti, Err(trap));
    }

    /// #926 slice 2e — deliver a surfaced `Jit.invoke`'s emitted `f0` result slots into the paused
    /// task and clear the pending slot (mirroring [`deliver_tierup`](Self::deliver_tierup), routed to
    /// the invoking task's frame via `pending_jit`). A short reply traps the task.
    fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        // #926 slice 2g: the emitted invoke resolved — its bounce registry dies with it (Vcpu parity).
        self.invoke_fibers.clear();
        let (ti, dst, results) = self
            .pending_jit
            .take()
            .expect("deliver_jit_invoke_vals with no pending invoke");
        if vals.len() < results.len() {
            complete(&mut self.tasks, ti, Err(Trap::Malformed));
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.tasks[ti].vt.active.set(
                dst as u32 + i as u32,
                Reg::from_value(slot_to_val(*ty, vals[i])),
            );
        }
    }

    /// #926 slice 2e — the emitted `Jit.invoke` unit trapped: trap the invoking task (mirroring
    /// [`deliver_tierup_trap`](Self::deliver_tierup_trap)).
    fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        // #926 slice 2g: the emitted invoke resolved (trapped) — its bounce registry dies with it.
        self.invoke_fibers.clear();
        let (ti, _dst, _results) = self
            .pending_jit
            .take()
            .expect("deliver_jit_invoke_trap with no pending invoke");
        complete(&mut self.tasks, ti, Err(trap));
    }
}

/// #926 slice 2 tier-up configuration for a cooperative run: the wasm-JIT eligibility bitmap plus
/// whether it is page-checked (#750). Bundled rather than passed as two loose parameters because
/// `page_checked` is meaningful only alongside a bitmap — a `None` config is "no tier-up", exactly the
/// native `drive`. The bitmap is per **module-0** function; a direct call to an `eligible[f] == true`
/// function surfaces as a tier-up on the root and any same-module `thread.spawn` descendant.
pub struct TierUpConfig {
    /// Per-module-0-function eligibility (index = function). `true` ⇒ a direct call tiers up.
    pub eligible: std::sync::Arc<[bool]>,
    /// #750 paged tier-up: the emitted region carries a per-access page check, so an unrepresentable
    /// window surfaces with the reserved size instead of declining.
    pub page_checked: bool,
}

/// A pause of the cooperative tier-up driver [`CoopRun`], mirroring the single-vCPU [`VcpuEvent`]'s
/// tier-up-relevant subset. The cooperative driver services concurrency (`thread.spawn`, join, futex
/// wait/notify) **internally** — multiplexing every vCPU on the one host thread — so, unlike the
/// per-Worker parallel driver, those never surface; only the run's end and tier-up round-trips do.
pub enum CoopEvent {
    /// The run finished; these are the root task's results.
    Done(Vec<Value>),
    /// The run trapped (the root task, or a fatal driver fault).
    Trapped(Trap),
    /// A module-0 task paused on an eligible direct `Call` to `func` with raw i64 arg slots `argv`;
    /// `mapped` is the committed scalar window extent for the emitted `"mapped"` global. The host runs
    /// the emitted `f{func}` and calls [`CoopRun::deliver_tierup`] / [`CoopRun::deliver_tierup_trap`].
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        mapped: u64,
    },
    /// A task paused on a §22 `Jit.invoke` of a runtime-compiled unit with emitted `wasm`: the host
    /// runs the unit's `f0(win, env, ...argv)` (marshalling by `params`/`results`, `mapped` into its
    /// `"mapped"` global) and calls [`CoopRun::deliver_jit_invoke_vals`] /
    /// [`CoopRun::deliver_jit_invoke_trap`]. A non-emittable unit runs interpreted and never surfaces.
    JitInvoke {
        code: i32,
        wasm: std::sync::Arc<[u8]>,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: u64,
    },
}

/// A **resumable cooperative tier-up run**: the single-thread, no-Worker analogue of the parallel
/// `svm_par_*` driver. It owns the run's `Domain`/window/powerbox/fuel and a [`CoopSched`] that
/// multiplexes every vCPU (root + `thread.spawn` descendants) on this one thread, and pauses to the
/// host on each wasm-JIT tier-up ([`run`](Self::run) → [`CoopEvent::TierUp`] → run the emitted region
/// → [`deliver_tierup`](Self::deliver_tierup) → `run` again), exactly the loop the native tests and
/// the browser cdylib drive it with. With no eligibility bitmap it behaves as `drive`: `run` returns
/// `Done`/`Trapped` in one call. (#926 slice 2.)
pub struct CoopRun {
    dom: Domain,
    mem: Option<Mem>,
    host: Host,
    fuel: u64,
    sched: CoopSched,
}

impl CoopRun {
    /// Build a run over module `m`, entering `entry(args)` with `fuel` and the granted `host` powerbox.
    /// `tierup` is the tier-up config ([`TierUpConfig`]); pass `None` for a pure-interpreter multiplex.
    /// `None` if `m` uses an op outside the bytecode engine's subset (the caller falls back to the
    /// tree-walker), `Some(Err)` if `entry` is out of range.
    pub fn new(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        tierup: Option<TierUpConfig>,
    ) -> Option<Result<CoopRun, Trap>> {
        // A fresh engine-sized window built from `m`'s declaration + data (the native/test path).
        Self::assemble(m, entry, args, fuel, host, tierup, build_mem(m))
    }

    /// Like [`new`](Self::new), but the linear-memory window is built **over a caller-provided
    /// backing** `back` (with `init_mem` seeded first), the resumable twin of
    /// [`Vcpu::new_root_reserved_over_with_powerbox`]. This is the browser cdylib seam: the window
    /// lives in the host's own linear memory, so every emitted `f{func}(win, env, …)` addresses it
    /// directly through the one shared `env.memory`. `back` is dropped if `m` is unsupported.
    #[allow(clippy::too_many_arguments)] // the window-backing seam inherently threads more inputs
    pub fn new_over(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        tierup: Option<TierUpConfig>,
        init_mem: &[u8],
        reserved_log2: u8,
        back: std::sync::Arc<super::Region>,
    ) -> Option<Result<CoopRun, Trap>> {
        let mem = m.memory.map(|mc| {
            let mut mm = Mem::with_reservation_over(reserved_log2, mc.size_log2, back);
            mm.seed(init_mem);
            mm.init_data(&m.data);
            mm.seed_null_guard(svm_ir::module_null_guard(m).unwrap_or(0)); // #964
            mm
        });
        Self::assemble(m, entry, args, fuel, host, tierup, mem)
    }

    /// Shared constructor tail: compile `m`, range-check `entry`, and build the `CoopSched` over the
    /// caller-chosen `mem`. `None` if `m` is outside the bytecode engine's subset (fall back to the
    /// tree-walker); `Some(Err)` if `entry` is out of range or seeding traps.
    fn assemble(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        mut fuel: u64,
        mut host: Host,
        tierup: Option<TierUpConfig>,
        mut mem: Option<Mem>,
    ) -> Option<Result<CoopRun, Trap>> {
        let c = compile_module_for(m)?;
        if entry as usize >= c.progs.len() {
            return Some(Err(Trap::Malformed));
        }
        let dom = Domain::new(c, host.jit_table_log2());
        let sched = match CoopSched::new(&dom, entry, args, &mut fuel, &mut mem, &mut host, tierup)
        {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        }))
    }

    /// The run's window committed **scalar extent** right now — the #717 value the cdylib re-syncs to
    /// every emitted instance's `"mapped"` global after a [`bounce`](Self::bounce) (a bounced callback
    /// may have grown the window). `0` when there is no window or its state is not representable by one
    /// bound. Reads the shared root window (a confined child's own-window growth is a later refinement).
    pub fn window_scalar_extent(&self) -> u64 {
        self.mem
            .as_ref()
            .and_then(|m| m.scalar_extent())
            .unwrap_or(0)
    }

    /// The run's **root** powerbox — where the root task and its `thread.spawn` threads' host I/O
    /// lands (stdout/stderr, the framebuffer). The cdylib drains it into its capture slots at the end
    /// of a run. (§14 confined children keep their own `host` in `extra_envs`, not exposed here.)
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// #926 slice 2f — the §22 code handle installed at dispatch-table `slot` (`-1` empty/natural):
    /// the browser B2 driver's slot mirror, from which it rebuilds its `WebAssembly.Table` at each
    /// event boundary (a slot at or past the program's `f{i}` prefix holds an installed unit's `f0`).
    pub fn slot_code(&self, slot: u32) -> i32 {
        self.sched
            .slot_codes
            .get(slot as usize)
            .copied()
            .unwrap_or(-1)
    }

    /// #1009: the dispatch-table generation — bumped on each `Jit.install`/`Jit.uninstall` the
    /// scheduler services. The browser B2 driver caches the generation it last synced its
    /// `WebAssembly.Table` at and rebuilds only when this advances (the single-shot pump's
    /// `svm_onramp_tierup_table_gen` twin).
    pub fn table_gen(&self) -> u32 {
        self.sched.table_gen
    }

    /// #1009 paged tier-up: the root window's memory-map introspection ([`MemMapInfo`]) — a paged
    /// coop driver rebuilds its page-state table from this. `None` for a memory-less run.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// #1009 paged tier-up: the root window's page-map version (bumped on every `map`/`unmap`/
    /// `protect`) — the cheap `O(1)` counter a paged coop driver compares to skip an unchanged
    /// page-state rebuild. `0` for a memory-less run.
    pub fn mem_map_version(&self) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.map_version())
    }

    /// Pump the schedule to its next pause: [`CoopEvent::Done`]/[`CoopEvent::Trapped`] end the run,
    /// [`CoopEvent::TierUp`] hands an emitted region to the host (resume with `deliver_tierup*`).
    pub fn run(&mut self) -> CoopEvent {
        // `budget` is the per-step op budget `step_vcpu` hands `Vm::resume` (a `0` would run zero ops
        // and spin on `Outcome::Suspended`); the native `drive` runs unsliced, so match it with
        // `u64::MAX` — run each vCPU to its next stop. It doubles as the §3d spawn record budget
        // (`u64::MAX` ⇒ the no-record default `take_spawn_budget` already uses for a normal run).
        match self.sched.pump(
            &self.dom,
            &mut self.mem,
            &mut self.host,
            &mut self.fuel,
            u64::MAX,
        ) {
            Ok(CoopStep::Done(vals)) => CoopEvent::Done(vals),
            Ok(CoopStep::TierUp { func, argv, mapped }) => CoopEvent::TierUp { func, argv, mapped },
            Ok(CoopStep::JitInvoke {
                code,
                wasm,
                argv,
                params,
                results,
                mapped,
            }) => CoopEvent::JitInvoke {
                code,
                wasm,
                argv,
                params,
                results,
                mapped,
            },
            Err(t) => CoopEvent::Trapped(t),
        }
    }

    /// Deliver the emitted tier-up region's raw i64 result slots and resume (see
    /// [`CoopSched::deliver_tierup`]). Call exactly once after a [`CoopEvent::TierUp`], before `run`.
    pub fn deliver_tierup(&mut self, vals: &[i64]) {
        self.sched.deliver_tierup(vals);
    }

    /// Surface an emitted tier-up region's trap and resume (the paused task traps). Call once after a
    /// [`CoopEvent::TierUp`] in lieu of [`deliver_tierup`](Self::deliver_tierup).
    pub fn deliver_tierup_trap(&mut self, trap: Trap) {
        self.sched.deliver_tierup_trap(trap);
    }

    /// Deliver a surfaced `Jit.invoke` unit's emitted `f0` result slots and resume (see
    /// [`CoopSched::deliver_jit_invoke_vals`]). Call once after a [`CoopEvent::JitInvoke`], before `run`.
    pub fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        self.sched.deliver_jit_invoke_vals(vals);
    }

    /// Surface a `Jit.invoke` unit's trap and resume (the invoking task traps). Call once after a
    /// [`CoopEvent::JitInvoke`] in lieu of [`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals).
    pub fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        self.sched.deliver_jit_invoke_trap(trap);
    }

    /// #926 slice 2 — service an emitted tier-up region's **or** a surfaced `Jit.invoke` unit's cross-tier
    /// `call_interp(target, io)` while it is mid-run for the **currently paused task**. Routes the bounce
    /// to *that task's* env — a §14 confined child steps its own window/powerbox/dispatch table
    /// (`env == Some`), the root and its `thread.spawn` threads the shared ones (`env == None`) — so a
    /// confined leaf's callback can never reach outside its window (the confinement hinge). The fiber
    /// registry is picked by which round-trip is outstanding (#926 slice 2g, `Vcpu::bounce_call` parity):
    /// a tier-up region uses the **run-level** registry (a parked fiber persists for the run to resume),
    /// while a surfaced `Jit.invoke` uses the **invoke-confined** registry (`invoke_fibers` — its fibers
    /// die when the invoke resolves). Marshals results back into `io` and returns the result count. Call
    /// only between a [`CoopEvent::TierUp`]/[`CoopEvent::JitInvoke`] and its delivery; `Err(Malformed)` if
    /// nothing is outstanding.
    pub fn bounce(&mut self, target: u32, io: &mut [i64]) -> Result<usize, Trap> {
        // The paused task is whichever host round-trip is outstanding — a tier-up region or a
        // surfaced `Jit.invoke` unit; both bounce cross-tier the same way. (#926 slice 2e)
        let ti = self
            .sched
            .pending_tierup
            .as_ref()
            .or(self.sched.pending_jit.as_ref())
            .map(|(ti, ..)| *ti)
            .ok_or(Trap::Malformed)?;
        // Mid-invoke iff a `Jit.invoke` (not a tier-up) is the outstanding round-trip — never both at
        // once (the one-round-trip discipline). Selects the invoke-confined registry below.
        let in_invoke = self.sched.pending_jit.is_some();
        let CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        } = self;
        let CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            invoke_fibers,
            ..
        } = sched;
        // The registry `coop_bounce` threads into `drive_nested`: invoke-confined (`invoke_fibers`, no
        // shadow-SP/freeze halves — invoke fibers are transient) during an emitted `Jit.invoke`, else the
        // run-level registry with its parallel arrays. One of the two `match` arms below moves it.
        let (bounce_fibers, bounce_meta): (&mut Vec<FiberState>, Option<RunFiberMeta<'_>>) =
            if in_invoke {
                (invoke_fibers, None)
            } else {
                (fibers, Some((fiber_sp, fiber_meta)))
            };
        match tasks[ti].env {
            // Root / `thread.spawn` thread: the run's shared window, powerbox, and domain table.
            None => {
                let mut cell = HostCell::Excl(host);
                coop_bounce(
                    &dom.source,
                    &dom.table,
                    fuel,
                    mem,
                    &mut cell,
                    bounce_fibers,
                    bounce_meta,
                    target,
                    io,
                )
            }
            // §14 confined child: its OWN window, powerbox, table, and fuel — never the root's.
            Some(k) => {
                let e = &mut extra_envs[k];
                let mut cell = HostCell::Shared(&e.host);
                coop_bounce(
                    &dom.source,
                    &e.table,
                    &mut e.fuel,
                    &mut e.mem,
                    &mut cell,
                    bounce_fibers,
                    bounce_meta,
                    target,
                    io,
                )
            }
        }
    }
}

/// #926 slice 2 — the cooperative driver's cross-tier **bounce** body: an emitted tier-up region, mid
/// run, calls back to an interp-resident leaf `target`. Resolves `target` through the caller's dispatch
/// `table` (masked, exactly as `Op::CallIndirect`), marshals the i64 scratch `io` per the callee's
/// signature, and drives it to completion on a nested interpretation over the caller's `mem`/`host`/
/// `fuel`. The `fibers`/`fiber_meta` the caller passes select the registry (#926 slice 2g): a **tier-up
/// region**'s callbacks run against the **run-level** registry (`fiber_meta = Some(shadow-SP/freeze
/// halves)` — a parked fiber persists for the run, the same registry `step_vcpu` mirrors its parallel
/// arrays into), while a surfaced **`Jit.invoke`** unit's callbacks run against an **invoke-confined**
/// registry (`fiber_meta = None`, the transient loop-local scope `run_invoke` and `Vcpu::bounce_call`'s
/// invoke branch use — its fibers die when the invoke resolves). The multi-task analogue of
/// [`Vcpu::bounce_call`], differing only in that the window/powerbox/table come from the *paused task's*
/// env (resolved by [`CoopRun::bounce`]).
#[allow(clippy::too_many_arguments)] // an inherently many-input dispatch shim; a config struct would obscure it
fn coop_bounce(
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
    fibers: &mut Vec<FiberState>,
    fiber_meta: Option<RunFiberMeta<'_>>,
    target: u32,
    io: &mut [i64],
) -> Result<usize, Trap> {
    step(fuel, None)?; // fuel unification: the dispatch-site safepoint
    let slot = (target as usize) & (table.len() - 1);
    let ts = table.slot(slot);
    if ts.module == super::TABLE_EMPTY {
        return Err(Trap::IndirectCallType);
    }
    let tm = source.get(ts.module as usize).ok_or(Trap::Malformed)?;
    let (cp, cr) = tm.sigs[ts.func as usize].clone();
    if cp.len() > io.len() || cr.len() > io.len() {
        return Err(Trap::Malformed); // scratch too small — a mis-marshalled host call
    }
    // The i64-slot transport carries scalars only; a v128-sig target can never have been given a
    // trampoline (the host gates that at open) — a bounce naming one is a mis-wired host.
    let scalar =
        |t: &ValType| matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64);
    if !cp.iter().all(scalar) || !cr.iter().all(scalar) {
        return Err(Trap::Malformed);
    }
    let args: Vec<Value> = cp
        .iter()
        .zip(io.iter())
        .map(|(ty, s)| slot_to_val(*ty, *s))
        .collect();
    let mut vm = Vm::new(&tm, ts.func as usize, &args)?;
    vm.module = ts.module as usize;
    let vals = drive_nested(source, table, vm, fuel, mem, host, fibers, fiber_meta)?;
    for (i, v) in vals.iter().enumerate() {
        io[i] = val_to_slot(*v);
    }
    Ok(cr.len())
}

/// THREADS.md step 4c — a **native futex**, the parallel driver's stand-in for wasm
/// `memory.atomic.wait`/`notify`. A parked waiter enqueues a token (its own `woken` flag + `Condvar`)
/// under its address key; `notify` wakes up to `count` of them FIFO. The compare-and-park runs under
/// `buckets`, so a concurrent `notify` cannot slip between a waiter reading the futex word and parking
/// (the std-sync analogue of the kernel's per-bucket futex lock) — no lost wakeups. In real wasm this
/// role is played by `memory.atomic.wait`/`notify` directly; here it serves the cooperative oracle's
/// same `wait`/`notify` semantics for genuinely parallel vCPUs.
#[derive(Default)]
struct Futex {
    buckets: std::sync::Mutex<
        std::collections::HashMap<u64, std::collections::VecDeque<std::sync::Arc<Waiter>>>,
    >,
}

struct Waiter {
    woken: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl Futex {
    /// `memory.wait`: compare the futex word at `base` to `expected` under the bucket lock; if it
    /// already differs, return `WAIT_NOT_EQUAL` without parking (the fast path). Otherwise enqueue a
    /// token and park on it until `notify` wakes it (`WAIT_WOKEN`) or `timeout` ns elapse
    /// (`WAIT_TIMED_OUT`). Mirrors the cooperative `BlockedWait` arm; the per-token flag absorbs
    /// spurious condvar wakeups.
    fn wait(&self, mem: &Mem, base: u64, expected: u64, width: u32, timeout: u64) -> i32 {
        let waiter = {
            let mut buckets = self.buckets.lock().unwrap();
            // Compare-under-lock: the futex word lives in the shared backing (`atomic_value` reads it).
            if mem.atomic_value(base, width) != expected {
                return super::WAIT_NOT_EQUAL;
            }
            let w = std::sync::Arc::new(Waiter {
                woken: std::sync::Mutex::new(false),
                cv: std::sync::Condvar::new(),
            });
            buckets
                .entry(base)
                .or_default()
                .push_back(std::sync::Arc::clone(&w));
            w
        };
        // Park on our own token (the bucket lock is released): woken by `notify`, or timed out.
        let timeout = std::time::Duration::from_nanos(timeout);
        let (flag, res) = waiter
            .cv
            .wait_timeout_while(waiter.woken.lock().unwrap(), timeout, |w| !*w)
            .unwrap();
        let woken = *flag;
        drop(flag);
        if woken {
            super::WAIT_WOKEN
        } else {
            debug_assert!(res.timed_out());
            // Timed out: de-enqueue our (possibly still-parked) token so a later `notify` skips it.
            let mut buckets = self.buckets.lock().unwrap();
            if let Some(q) = buckets.get_mut(&base) {
                q.retain(|x| !std::sync::Arc::ptr_eq(x, &waiter));
            }
            super::WAIT_TIMED_OUT
        }
    }

    /// `memory.notify`: wake up to `count` waiters parked on `base`, FIFO, and return how many were
    /// woken (mirrors the cooperative `Notify` arm's count; the guest typically ignores it).
    fn notify(&self, base: u64, count: i32) -> i32 {
        let want = count as u32;
        let mut buckets = self.buckets.lock().unwrap();
        let mut woken = 0u32;
        if let Some(q) = buckets.get_mut(&base) {
            while woken < want {
                let Some(w) = q.pop_front() else { break };
                *w.woken.lock().unwrap() = true;
                w.cv.notify_one();
                woken += 1;
            }
        }
        woken as i32
    }
}

/// THREADS.md step 4c — the cross-thread `thread.spawn`/`join` rendezvous for the parallel driver.
/// The cooperative `drive` keeps its child vCPUs in one `tasks` vec and wakes joiners inline; the
/// parallel driver runs each vCPU on its **own OS thread**, so a joiner blocks here on a `Condvar`
/// until the child it named publishes its result. One `id` namespace across the whole run (handed out
/// by `next_id`); a child's result (value-or-trap) is delivered to the lowest-index waiter via the
/// `done` map. `live` mirrors the cooperative `MAX_VCPUS` anti-bomb gate across threads. `futex` serves
/// the guest's `memory.wait`/`notify` across threads.
struct ThreadRegistry {
    done: std::sync::Mutex<std::collections::HashMap<u64, Result<Vec<Value>, Trap>>>,
    woken: std::sync::Condvar,
    next_id: std::sync::atomic::AtomicU64,
    live: std::sync::atomic::AtomicUsize,
    futex: Futex,
}

impl ThreadRegistry {
    fn new() -> ThreadRegistry {
        ThreadRegistry {
            done: std::sync::Mutex::new(std::collections::HashMap::new()),
            woken: std::sync::Condvar::new(),
            next_id: std::sync::atomic::AtomicU64::new(0),
            live: std::sync::atomic::AtomicUsize::new(0),
            futex: Futex::default(),
        }
    }

    /// A spawned vCPU finished: publish its result and wake any joiner parked on it.
    fn publish(&self, id: u64, res: Result<Vec<Value>, Trap>) {
        self.done.lock().unwrap().insert(id, res);
        self.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        self.woken.notify_all();
    }

    /// Block until vCPU `id` has published, then take (consume) its result — the parallel analogue of
    /// the cooperative `BlockedJoin` wakeup. A child trap is returned to propagate to the joiner.
    fn join(&self, id: u64) -> Result<Vec<Value>, Trap> {
        let mut g = self.done.lock().unwrap();
        loop {
            if let Some(r) = g.remove(&id) {
                return r;
            }
            g = self.woken.wait(g).unwrap();
        }
    }
}

/// THREADS.md step 4c — the **parallel** driver (the host-selected `Parallel` mode). One guest's vCPUs
/// run on **separate OS threads** sharing **one** `Region::shared` window, instead of the cooperative
/// `drive`'s single-thread `tasks` loop. `std::thread::scope` borrows the `&Domain` (which is `Sync`)
/// and the `&ThreadRegistry` into each child and joins every still-running thread before returning, so
/// the window is quiescent for the snapshot. The root runs on the calling thread (it never
/// `atomic.wait`s — `join` blocks on a `Condvar`, sidestepping the browser main-thread-wait wrinkle).
/// Returns the root's result and its (now-quiescent) `Mem` for capture. Scope: the pure-threads subset
/// (`thread.spawn`/`join` + atomics); other multi-vCPU events fail closed (see
/// [`compile_and_run_capture_over_parallel`]).
fn drive_parallel(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: u64,
    mem: Option<Mem>,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Option<Mem>) {
    let root_vt = match VTask::new(&dom.source.primary(), entry as usize, args) {
        Ok(v) => v,
        Err(t) => return (Err(t), mem),
    };
    let reg = ThreadRegistry::new();
    // Share the caller's powerbox across every vCPU thread, then hand it back (so the caller reads its
    // stdout / final state). `scope` joins all vCPUs before returning, so the borrow is sound and the
    // `Mutex` is uncontended at unwrap.
    let shared = std::sync::Mutex::new(std::mem::take(host));
    let out = std::thread::scope(|scope| {
        run_vcpu_parallel(scope, &dom, &reg, &shared, root_vt, mem, fuel)
    });
    *host = shared.into_inner().unwrap_or_else(|e| e.into_inner());
    out
}

/// Run one vCPU of the parallel driver to completion on **this** OS thread, fanning each
/// `thread.spawn` onto a fresh scoped thread (over a `fork_for_thread` view of the shared window) and
/// blocking each `thread.join` on the [`ThreadRegistry`]. Mirrors the cooperative `drive`'s `Spawn` /
/// `Join` / `Done` arms, one vCPU at a time. Returns this vCPU's result and the `Mem` it owned (the
/// root's is the one captured; a child's is dropped, its bytes already live in the shared backing).
fn run_vcpu_parallel<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    dom: &'env Domain,
    reg: &'env ThreadRegistry,
    host: &'env std::sync::Mutex<Host>,
    mut vt: VTask,
    mut mem: Option<Mem>,
    mut fuel: u64,
) -> (Result<Vec<Value>, Trap>, Option<Mem>) {
    let mut fibers: Vec<FiberState> = Vec::new();
    let mut fiber_sp: Vec<u64> = Vec::new();
    let mut fiber_meta: Vec<(i32, i64)> = Vec::new();
    // handle (index) → global vCPU id of a `thread.spawn` child (shares the cooperative handle scheme).
    let mut threads: Vec<Option<u64>> = Vec::new();
    loop {
        let mut ctx = RunCtx {
            table: &dom.table,
            fuel: &mut fuel,
            mem: &mut mem,
            durable: false,
            // The powerbox is **shared** by every vCPU of the run (4c-host): `cap.call` takes the lock
            // only for its own dispatch, so compute/atomics/futex between calls stay lock-free.
            host: HostCell::Shared(host),
        };
        // NLL ends `ctx`'s borrows of `mem`/`fuel` at this call, so the arms below may touch them.
        let stop = step_vcpu(
            &mut vt,
            &mut fibers,
            &mut fiber_sp,
            &mut fiber_meta,
            dom,
            &mut ctx,
            u64::MAX,
            false, // OS-thread parallel driver: blocking `cont.resume.block` idle is a follow-up (I48)
        );
        match stop {
            // §3.6 (I36 slice 2): the serve/call/offer trio runs only on the cooperative
            // driver (`drive`); a serving module never reaches the parallel driver (the
            // qualification veto refuses svc + threads together) — fail closed if it somehow
            // does, rather than park unwakeably. I48 `BlockOnFiber` is likewise cooperative-only
            // (this path passes `cooperative: false`), so it never arises here — grouped in.
            Ok(VcpuStop::LiveCall { .. })
            | Ok(VcpuStop::SvcWait)
            | Ok(VcpuStop::ChildOffer { .. })
            | Ok(VcpuStop::CloneCaller { .. })
            | Ok(VcpuStop::Reap { .. })
            | Ok(VcpuStop::BlockOnFiber { .. }) => return (Err(Trap::ThreadFault), mem),
            Err(trap) => return (Err(trap), mem),
            Ok(VcpuStop::Done(vals)) => return (Ok(vals), mem),
            // Tier-up is only enabled on the browser `Vcpu::run` path (`with_jit_eligible`).
            Ok(VcpuStop::TierUp { .. }) => unreachable!("tier-up not enabled on the native driver"),
            // Blocking stdin is only ever set on an owned-host `Vcpu` (the interactive session).
            Ok(VcpuStop::StdinPark) => {
                unreachable!("blocking stdin not enabled on the native driver")
            }
            Ok(VcpuStop::Spawn {
                func,
                sp,
                arg,
                dst,
                module,
            }) => {
                // Module-aware, as the cooperative arm: `func` is the spawning frame's module's index.
                let Some(cm) = dom.source.get(module as usize) else {
                    return (Err(Trap::Malformed), mem);
                };
                if func as usize >= cm.progs.len() {
                    return (Err(Trap::Malformed), mem);
                }
                // Cross-thread anti-bomb gate (mirrors the cooperative `live >= MAX_VCPUS`).
                if reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
                    > super::MAX_VCPUS
                {
                    reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return (Err(Trap::ThreadFault), mem);
                }
                let id = reg
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let child_vt =
                    match VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)]) {
                        Ok(mut v) => {
                            v.active.module = module as usize;
                            v.active.home = module as usize;
                            v
                        }
                        Err(t) => return (Err(t), mem),
                    };
                // The child runs over its own `Mem` view of the **same** shared backing (real atomics).
                let child_mem = mem.as_ref().map(|m| m.fork_for_thread());
                scope.spawn(move || {
                    let (r, _m) =
                        run_vcpu_parallel(scope, dom, reg, host, child_vt, child_mem, fuel);
                    reg.publish(id, r);
                });
                let handle = threads.len() as i32;
                threads.push(Some(id));
                vt.active.set(dst, Reg::from_i32(handle));
            }
            Ok(VcpuStop::Join { handle, dst }) => {
                let slot = match super::resolve_thread(&threads, handle) {
                    Ok(s) => s,
                    Err(t) => return (Err(t), mem),
                };
                let id = threads[slot].expect("resolve_thread checked liveness");
                threads[slot] = None; // single join — the handle is now spent
                match reg.join(id) {
                    // A joined child's first result value lands in the joiner's `dst`.
                    Ok(vals) => {
                        let v = vals.first().copied().unwrap_or(Value::I64(0));
                        vt.active.set(dst, Reg::from_value(v));
                    }
                    // A child trap propagates: the joiner completes with the same trap.
                    Err(t) => return (Err(t), mem),
                }
            }
            Ok(VcpuStop::CapPending { id, dst }) => {
                // F2: the parallel driver keeps the inline completion wait — it blocks only
                // this OS thread while the pool works (the §5b overlap across sibling vCPUs
                // is the lock-release, already landed); fiber-waiter delivery through the
                // real cross-thread futex is the I45/I73 residue, its own slice.
                let comps = host.lock_unpoisoned().completions();
                let r = comps.wait(id);
                vt.active.set(dst, Reg::from_i64(r));
            }
            Ok(VcpuStop::Wait {
                base,
                expected,
                width,
                timeout,
                dst,
            }) => {
                // Genuine cross-thread futex: park on the shared address until another vCPU `notify`s
                // (or the timeout fires). No memory ⇒ can't park ⇒ vacuously not-equal.
                let r = match mem.as_ref() {
                    Some(m) => reg.futex.wait(m, base, expected, width, timeout),
                    None => super::WAIT_NOT_EQUAL,
                };
                vt.active.set(dst, Reg::from_i32(r));
            }
            Ok(VcpuStop::Notify { base, count, dst }) => {
                let woken = reg.futex.notify(base, count);
                vt.active.set(dst, Reg::from_i32(woken));
            }
            // §22 guest-JIT (THREADS.md 4c-domain): install/uninstall/invoke against the **shared**
            // [`Domain`] — `install`/`uninstall`/`push` are interior-mutable (Release/Acquire-paired
            // with the dispatch reads), so a worker vCPU drives them on `&Domain` while compute/atomics
            // on the other vCPUs stay lock-free. The result (slot / `-ENOSPC` / value) is
            // schedule-independent for the disciplined guest the oracle is differentially run against.
            Ok(VcpuStop::JitInstall { h, code, dst }) => {
                // Resolve authority + the unit's funcs under the host lock (a forged/cross-domain
                // handle is an inert CapFault → trap), then compile + install. Compiling can fail only
                // if the unit uses an op the engine doesn't lower yet (the one place a guest unit can
                // outrun coverage — no tree-walker fallback mid-run).
                let funcs = {
                    let g = host.lock_unpoisoned();
                    match g.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = g.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        g.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
                    }) {
                        Ok(f) => f,
                        Err(t) => return (Err(t), mem),
                    }
                };
                let res = match compile_module(&funcs) {
                    Some(unit) => match dom.install(unit) {
                        Some(slot) => slot as i64,
                        None => super::ENOSPC,
                    },
                    None => return (Err(Trap::Malformed), mem), // unit op outside coverage
                };
                vt.active.set(dst, Reg::from_i64(res));
            }
            Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                {
                    let g = host.lock_unpoisoned();
                    if let Err(t) = g.resolve_jit_domain(h) {
                        return (Err(t), mem); // authority check
                    }
                }
                let n_real = dom.source.primary().progs.len();
                let res = if dom.uninstall(slot as usize, n_real) {
                    0
                } else {
                    super::EINVAL
                };
                vt.active.set(dst, Reg::from_i64(res));
            }
            Ok(VcpuStop::JitInvoke {
                h,
                code,
                argv,
                dst,
                params,
                results,
            }) => {
                // Resolve unit funcs (authority + cross-domain) and compile, as for install.
                let funcs = {
                    let g = host.lock_unpoisoned();
                    match g.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = g.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        g.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
                    }) {
                        Ok(f) => f,
                        Err(t) => return (Err(t), mem),
                    }
                };
                let unit = match compile_module(&funcs) {
                    Some(u) => u,
                    None => return (Err(Trap::Malformed), mem),
                };
                // Arity-check the unit entry (func 0) against the call's (code-stripped) signature.
                let arity_ok = unit
                    .sigs
                    .first()
                    .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
                if !arity_ok {
                    return (Err(Trap::CapFault), mem);
                }
                // Marshal args via the slot ABI, push the unit as a transient module, run it over the
                // **shared** powerbox (its `cap.call`s serialize per-call, like every other vCPU's).
                let child_args: Vec<Value> = params
                    .iter()
                    .zip(argv.iter())
                    .map(|(ty, s)| slot_to_val(*ty, *s))
                    .collect();
                let umod = dom.source.push(unit);
                match run_invoke(
                    &dom.source,
                    &dom.table,
                    umod,
                    &child_args,
                    &mut fuel,
                    &mut mem,
                    &mut HostCell::Shared(host),
                ) {
                    Ok(vals) => {
                        for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                            let re = slot_to_val(*ty, val_to_slot(*v));
                            vt.active.set(dst + i as u32, Reg::from_value(re));
                        }
                    }
                    Err(t) => return (Err(t), mem),
                }
            }
            // §14 `Instantiator.instantiate` (THREADS.md 4c-domain) — a **same-module** confined
            // executor child: its own power-of-two sub-window (`nested_view` of the shared backing,
            // own page-prot map), its own attenuated powerbox (`Instantiator` + `AddressSpace` over
            // `[0, child_size)`), its own natural dispatch table (no parent install slots), and a
            // quota sub-allocated from the parent's fuel. The child is a **nested confined parallel
            // run** on its own scoped thread — joinable through the parent's registry exactly like a
            // `thread.spawn` child. Unlike a `thread.spawn` child (which shares this vCPU's `Mem`
            // view + the shared powerbox), it owns all of these — the §14 confinement.
            Ok(VcpuStop::Instantiate {
                ibase,
                isize: isz,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            }) => {
                // op 11 (named-grant spawn) / a §3d budget record is driven only by the cooperative
                // single-thread `drive` path (the browser's wasm-safe entry); the OS-thread parallel
                // driver declines them.
                if grants.is_some() || budget != 0 {
                    return (Err(Trap::Malformed), mem);
                }
                // Validate the child entry signature against module 0 and the power-of-two-aligned
                // carve within `[0, isize)` — identical to the cooperative `drive` arm.
                let c0 = dom.source.primary();
                let want_as = c0
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                let ok_entry = c0
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, r)| child_entry_ok(p, r));
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = carve_fits(
                    off_u,
                    size_log2,
                    isz,
                    ibase,
                    mem.as_ref().map_or(0, |m| m.null_guard),
                );
                if !ok_entry || !fits {
                    vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    continue;
                }
                // Cross-thread anti-bomb gate (mirrors the cooperative `live >= MAX_VCPUS`).
                if reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
                    > super::MAX_VCPUS
                {
                    reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return (Err(Trap::ThreadFault), mem);
                }
                // This vCPU's own `mem`/`fuel` *are* its environment (no `extra_envs` indirection —
                // a confined parent already runs on its own thread with its own confined view), so
                // holder-relative `ibase`/`off` compose straight onto the backing-absolute base.
                let pbase = mem.as_ref().map_or(0, |m| m.window.base());
                let abs_base = pbase + ibase + off_u;
                let child_mem = mem
                    .as_ref()
                    .map(|m| m.nested_view(abs_base, size_log2 as u8));
                let mut child_host = Host::new();
                let cinst = child_host.grant_instantiator(0, child_size);
                let cas = child_host.grant_address_space(0, child_size);
                let child_args = if want_as {
                    vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                } else {
                    vec![Value::I64(cinst as i64)]
                };
                let child_fuel = if quota <= 0 {
                    fuel
                } else {
                    (quota as u64).min(fuel)
                };
                // Own table over the **shared** source (module 0 = the same primary the child runs).
                let child_table = build_table(c0.progs.len(), 0);
                let child_dom = Domain::child(std::sync::Arc::clone(&dom.source), child_table);
                let child_vt = match VTask::new(&c0, entry as usize, &child_args) {
                    Ok(v) => v,
                    Err(t) => return (Err(t), mem),
                };
                let id = reg
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                scope.spawn(move || {
                    // A confined nested run: the child owns its domain (own table, shared source
                    // `Arc`), its attenuated powerbox (`Excl`), its `nested_view` window, its quota,
                    // and its **own** thread registry (for threads/instantiates *it* spawns). Its
                    // result is published to the **parent's** `reg` so the parent's `join` finds it.
                    let child_reg = ThreadRegistry::new();
                    let child_host = std::sync::Mutex::new(child_host);
                    let (r, _m) = std::thread::scope(|cscope| {
                        run_vcpu_parallel(
                            cscope,
                            &child_dom,
                            &child_reg,
                            &child_host,
                            child_vt,
                            child_mem,
                            child_fuel,
                        )
                    });
                    reg.publish(id, r);
                });
                let handle = threads.len() as i32;
                threads.push(Some(id));
                vt.active.set(dst, Reg::from_i32(handle));
            }
            // §14 `Instantiator.instantiate_module` (THREADS.md 4c-domain) — a **separate-module**
            // confined child: the host (which holds the powerbox) is locked to resolve + clone the
            // granted `Module`, it is compiled to bytecode and **pushed to the shared source** (so it
            // resolves by index, like a `Jit.invoke` transient), the child's data segments are
            // materialized into the carve, and the child runs over its own table mapping into *its*
            // pushed module index. Everything else (confined window, attenuated powerbox, quota, own
            // registry, nested scoped thread, join) is exactly as op 0.
            Ok(VcpuStop::InstantiateModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                quota,
                dst,
                grants,
                budget,
            }) => {
                // op 13 (named-grant spawn) / a §3d budget record is driven only by the cooperative
                // single-thread `drive` path (the browser's wasm-safe entry); the OS-thread parallel
                // driver declines them.
                if grants.is_some() || budget != 0 {
                    return (Err(Trap::Malformed), mem);
                }
                // Resolve + clone the granted module under the host lock (a forged/closed/wrong-type
                // handle is an inert CapFault → trap).
                let (cfuncs, cmem_log2, cdata) = {
                    let g = host.lock_unpoisoned();
                    match g.resolve_module(mh) {
                        Ok(grant) => (grant.funcs.clone(), grant.memory_log2, grant.data.clone()),
                        Err(t) => return (Err(t), mem),
                    }
                };
                // Compile to bytecode — a module using an op the engine can't lower is the one place a
                // guest-provided program outruns coverage (a `Malformed` trap, as for `Jit.install`).
                let child_compiled = match compile_module(&cfuncs) {
                    Some(c) => c,
                    None => return (Err(Trap::Malformed), mem),
                };
                // Validate the entry against the *child module* and the carve; a separate-module
                // child's carve must equal its declared memory (§14 transparency).
                let want_as = child_compiled
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                let ok_entry = child_compiled
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, r)| child_entry_ok(p, r));
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = carve_fits(
                    off_u,
                    size_log2,
                    isz,
                    ibase,
                    mem.as_ref().map_or(0, |m| m.null_guard),
                );
                let mod_ok = cmem_log2 == Some(size_log2 as u8);
                if !ok_entry || !fits || !mod_ok {
                    vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    continue;
                }
                if reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
                    > super::MAX_VCPUS
                {
                    reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return (Err(Trap::ThreadFault), mem);
                }
                let pbase = mem.as_ref().map_or(0, |m| m.window.base());
                let abs_base = pbase + ibase + off_u;
                // Materialize the module's data segments into the carve *before* spawning the child
                // (the write happens-before the child thread, so it sees them), then the confined view.
                let child_mem = {
                    if let Some(m) = mem.as_ref() {
                        for d in cdata.iter() {
                            if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
                                for (k, &b) in d.bytes.iter().enumerate() {
                                    m.set_byte(abs_base + d.offset + k as u64, b);
                                }
                            }
                        }
                    }
                    mem.as_ref()
                        .map(|m| m.nested_view(abs_base, size_log2 as u8))
                };
                let mut child_host = Host::new();
                let cinst = child_host.grant_instantiator(0, child_size);
                let cas = child_host.grant_address_space(0, child_size);
                let child_args = if want_as {
                    vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                } else {
                    vec![Value::I64(cinst as i64)]
                };
                let child_fuel = if quota <= 0 {
                    fuel
                } else {
                    (quota as u64).min(fuel)
                };
                // Push the compiled module to the **shared** source and run the child over its own
                // table mapping into *its* module index (no parent install slots).
                let progs_len = child_compiled.progs.len();
                let cm = dom.source.push(child_compiled);
                let child_table = build_table_for(progs_len, 0, cm as u32);
                let child_dom = Domain::child(std::sync::Arc::clone(&dom.source), child_table);
                let cunit = match child_dom.source.get(cm) {
                    Some(u) => u,
                    None => return (Err(Trap::Malformed), mem),
                };
                let mut child_vt = match VTask::new(&cunit, entry as usize, &child_args) {
                    Ok(v) => v,
                    Err(t) => return (Err(t), mem),
                };
                child_vt.active.module = cm;
                child_vt.active.home = cm;
                let id = reg
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                scope.spawn(move || {
                    let child_reg = ThreadRegistry::new();
                    let child_host = std::sync::Mutex::new(child_host);
                    let (r, _m) = std::thread::scope(|cscope| {
                        run_vcpu_parallel(
                            cscope,
                            &child_dom,
                            &child_reg,
                            &child_host,
                            child_vt,
                            child_mem,
                            child_fuel,
                        )
                    });
                    reg.publish(id, r);
                });
                let handle = threads.len() as i32;
                threads.push(Some(id));
                vt.active.set(dst, Reg::from_i32(handle));
            }
        }
    }
}

/// Mark task `ti` finished with `res`, then wake any vCPU parked on `thread.join` of it: an `Ok`
/// result is delivered into the joiner's `dst` (it becomes runnable); a trap propagates — the joiner
/// completes with the same trap (transitively, via the worklist).
fn complete(tasks: &mut [TaskSlot], ti: usize, res: Result<Vec<Value>, Trap>) {
    let mut work = vec![(ti, res)];
    while let Some((done, res)) = work.pop() {
        tasks[done].state = TaskState::Done(res.clone());
        for (j, t) in tasks.iter_mut().enumerate() {
            let TaskState::BlockedJoin { child, slot, dst } = t.state else {
                continue;
            };
            if child != done {
                continue;
            }
            t.threads[slot] = None;
            match &res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    t.vt.active.set(dst, Reg::from_value(v));
                    t.state = TaskState::Runnable;
                }
                Err(trap) => work.push((j, Err(trap.clone()))),
            }
        }
    }
}

/// Domain lifetime & teardown, cooperative-bytecode form (DESIGN.md §12, owner 2026-07-24;
/// ISSUES.md I37): a member's trap/exit is terminal for its whole **domain** — the shared-window
/// world its `env` names (`None` = the root + its `thread.spawn` threads; `Some(k)` = a §14
/// child + its threads). Fixpoint: find a domain with a `Done(Err)` member and a still-live
/// member, kill every live member with the same trap (via [`complete`], so a cross-domain joiner
/// re-raises and a `poll` reports status 2 — the I37 supervision mechanics), then errno-wake
/// cross-domain callers parked through the dying child (D37 death-is-revocation: cancellation is
/// a value, never a hang); repeat until no such domain remains (a kill that propagates into a
/// joiner may fell *its* domain next). The root domain's death is read by the caller's loop-top
/// root check — a sibling's trap becomes the run's result. Runs before anything is scheduled, so
/// teardown is "next safepoint" prompt in the cooperative model.
fn teardown_domains(
    tasks: &mut [TaskSlot],
    extra_envs: &[ChildEnv],
    dead_envs: &mut std::collections::BTreeSet<usize>,
) {
    loop {
        let hit = tasks.iter().find_map(|t| {
            if let TaskState::Done(Err(trap)) = &t.state {
                match t.env {
                    // A child domain is processed exactly once (`dead_envs` is the marker), even
                    // when the trapping member was its only vCPU — a later call through it must
                    // still find it dead (errno, not a deadlock).
                    Some(k) if !dead_envs.contains(&k) => {
                        return Some((Some(k), trap.clone()));
                    }
                    // The root domain: a live member left means the sweep hasn't run yet (once
                    // every member is Done the caller's root check ends the run).
                    None if tasks
                        .iter()
                        .any(|u| u.env.is_none() && !matches!(u.state, TaskState::Done(_))) =>
                    {
                        return Some((None, trap.clone()));
                    }
                    _ => {}
                }
            }
            None
        });
        let Some((env, trap)) = hit else { return };
        if let Some(k) = env {
            dead_envs.insert(k);
        }
        for i in 0..tasks.len() {
            if tasks[i].env == env && !matches!(tasks[i].state, TaskState::Done(_)) {
                complete(tasks, i, Err(trap.clone()));
            }
        }
        // The dying child's undelivered dispatches: wake every caller parked on a ticket
        // against its host with the probeable errno (queued or admitted — no reply will ever
        // come), and drop the queue. Later calls are refused at the LiveCall arm (`dead_envs`).
        if let Some(k) = env {
            let dying = &extra_envs[k].host;
            for t in tasks.iter_mut() {
                if let TaskState::BlockedTicket { callee, dst, .. } = &t.state {
                    if std::sync::Arc::ptr_eq(callee, dying) {
                        let dst = *dst;
                        t.vt.active.set(dst, Reg::from_i64(super::CAP_REVOKED));
                        t.state = TaskState::Runnable;
                    }
                }
            }
            dying.lock_unpoisoned().svc_queue.clear();
        }
    }
}

/// The reified bytecode continuation — everything a suspended activation needs to resume, held as
/// an explicit value rather than on the host Rust call stack. The register file (`regs`), the stack
/// of suspended caller activations (`stack`), and the `(cur, base, pc)` cursor together fully
/// describe a paused vCPU: the flat analogue of the tree-walker's `Vec<Frame>`.
///
/// Holding the continuation as data (not as live host-stack frames) is the structural prerequisite
/// for the scheduler / fiber / thread / debug seams (INTERP_PERF.md Slice 1c): a later slice breaks
/// [`Vm::resume`]'s loop at suspension points (preemption budget, blocking op, debug stop), persists
/// the cursor back into `self`, and hands this struct to the caller to park / hash / resume — exactly
/// what `park_suspended(frames)` does for the tree-walker today.
/// A `<setjmp.h>` checkpoint (see [`Vm::setjmp_points`]): everything needed to re-enter a `setjmp`
/// activation. `longjmp` truncates [`Vm::stack`] to `depth` (the intervening activations discarded —
/// C has no cleanups), restores the `(module, cur, base, pc)` cursor, and sets the `dst` register to
/// the long-jump value. The activation's register window survives in place, so it is not snapshotted.
#[derive(Clone, Copy)]
struct ByteSetJmp {
    /// `Vm::stack` length at `setjmp` (the `setjmp` activation is the current one, not yet pushed).
    depth: usize,
    module: usize,
    cur: usize,
    base: usize,
    /// The op index just after the `setjmp`.
    pc: usize,
    /// The `setjmp` result's window slot (relative to `base`) — set to the long-jump value on re-entry.
    dst: u32,
}

// `Clone` for time-travel checkpointing (DEBUGGING.md W1): a single-vCPU `DebugRun` snapshots its
// active `Vm` into the `seek` checkpoint ladder. Every field is a plain value or read-only `Arc`
// (`jit_eligible` — the tier-up bitmap, shared not mutated), so this is a faithful deep copy; the
// guest page store is **not** here (it lives in `DebugRun::mem`, snapshotted separately via
// `Mem::window_snapshot`), so cloning a `Vm` never aliases another run's memory.
#[derive(Clone)]
struct Vm {
    /// Function-wide register file, shared across activations by register windows (`[base, base +
    /// nslots)` per activation). Grows on demand as calls open deeper windows.
    regs: Vec<Reg>,
    /// Suspended caller activations: `(module, prog, base, resume pc, absolute first result slot)`.
    /// `module` is carried so a cross-module `call_indirect` (into an installed §22 unit) returns to
    /// the caller's module.
    stack: Vec<(usize, usize, usize, usize, usize)>,
    /// The running activation's module (index into `Domain::mods`; 0 = primary), function index,
    /// window base, and op cursor.
    module: usize,
    cur: usize,
    base: usize,
    pc: usize,
    /// Edge-copy staging buffer (parallel-copy safety); kept here so it is reused across resumes.
    scratch: Vec<Reg>,
    /// `<setjmp.h>` checkpoints — `setjmp` records its activation's resume point here keyed by the
    /// guest `jmp_buf` address; `longjmp` looks it up. No register snapshot is needed (unlike the
    /// tree-walker): the flat per-function register layout gives each block its own slots, so the
    /// `setjmp` block's values survive a deeper call in place. Keyed by address (re-`setjmp` overwrites).
    setjmp_points: std::collections::BTreeMap<u64, ByteSetJmp>,
    /// §12.8 4A.5: the window offset of this context's shadow-SP **word** — the base of its own region
    /// (`shadow_region_base`), which `durable.shadow_base` returns so the instrumented IR addresses its
    /// per-context SP word. The root's is context 0 (`SHADOW_BASE`); a fiber's its `slot + 1`. Set when
    /// the Vm is created (fiber) / activated; unused on a non-durable run.
    durable_region_base: u64,
    /// **wasm-JIT tier-up bitmap** (browser wasm-JIT threads slice), for module-0 functions only. Set
    /// on the root Vm via [`Vcpu::with_jit_eligible`]; a direct `Call` in module 0 to an eligible
    /// function surfaces [`Outcome::TierUp`] instead of interpreting. `None` (fibers, invoked units,
    /// non-JIT runs) ⇒ everything interprets — tier-up is a pure acceleration, never a correctness gate.
    jit_eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 **paged tier-up**: the eligible functions were emitted with the software page-check
    /// (`compile_module_tierup_paged`), so the dispatch must NOT decline tier-up on a
    /// scalar-unrepresentable window — the host-maintained page table carries per-page fidelity,
    /// and the event's `mapped` becomes the reserved window size (the bound must never under-admit
    /// a table-admitted page). Set only via [`Vcpu::with_jit_page_checked`].
    jit_page_checked: bool,
    /// §3.6 serve-loop core (I36 slice 1): the in-flight handler's completion ticket — `Some`
    /// between admitting a handler activation (whose return linkage rewinds into the `SvcPoll`
    /// op) and the re-execution that settles its result — and the count of dispatches completed
    /// by the current `svc.poll` activation.
    serve_ticket: Option<u64>,
    serve_count: i64,
    /// §12 per-vCPU **thread-local register** (`vcpu.tls.get`/`set`). One i64 of per-vCPU state,
    /// seeded to this vCPU's dense id at construction (root = 0; a spawned thread's `Vm` is re-seeded
    /// to its id in `drive`'s `Spawn` arm), guest-overwritable. Read at the op's execution point.
    /// Mirrors the tree-walker's `Vm::tls`. (Multi-OS-thread fiber *migration* re-seeding — a fiber
    /// resumed on a different worker reading that worker's word — is a follow-up; the browser tier is
    /// single-OS-thread cooperative, where the sole worker is 0, so every read is a faithful `0`.)
    tls: i64,
    /// The domain's **home module** — the unit whose functions are its service handlers (0 for the
    /// primary; a separate-module child's pushed unit index). `svc.poll`/`svc.wait` only dispatch
    /// handlers while executing in this module: `svc_handler_func` resolves indices against the
    /// domain's registered `self_module`, so serving from any *other* unit (an installed §22 unit
    /// running in the root domain) would index the wrong program table — fail closed instead.
    home: usize,
}

impl Vm {
    /// Open the entry activation: a zero-based window sized to the entry function, seeded with the
    /// call arguments. Total — an out-of-range entry or arg overflow is a clean `Malformed` trap.
    /// Every entry (root, fiber, thread, coroutine) starts in module 0.
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<Vm, Trap> {
        let prog = c.progs.get(entry).ok_or(Trap::Malformed)?;
        let mut regs: Vec<Reg> = vec![Reg::default(); prog.nslots as usize];
        for (i, a) in args.iter().enumerate() {
            *regs.get_mut(i).ok_or(Trap::Malformed)? = Reg::from_value(*a);
        }
        Ok(Vm {
            regs,
            stack: Vec::new(),
            module: 0,
            cur: entry,
            base: 0,
            pc: 0,
            scratch: Vec::new(),
            setjmp_points: std::collections::BTreeMap::new(),
            durable_region_base: super::shadow_region_base(0), // root context (overwritten for fibers)
            jit_eligible: None, // set only on the root Vm via `Vcpu::with_jit_eligible`
            jit_page_checked: false,
            serve_ticket: None,
            serve_count: 0,
            tls: 0, // §12 per-vCPU TLS seed: dense vCPU id (root = 0; a spawned thread re-seeds to its id)
            home: 0,
        })
    }

    /// Write a value to a frame-relative slot of the *current* (persisted) activation window. Used
    /// by [`drive`] to deliver fiber results (`cont.new` handle, `cont.resume` `(status, value)`,
    /// the next `arg` into a `suspend`) into a `Vm` paused at a fiber op — `base` is the cursor the
    /// last `resume` persisted, so this targets the same window the op's `dst` was resolved against.
    fn set(&mut self, slot: u32, v: Reg) {
        self.regs[self.base + slot as usize] = v;
    }

    /// The [`crate::IrPc`] of the op the cursor is on, or `None` if that op is a terminator (which the
    /// debug seam never stops at — see [`Program::src`]). Used by [`ir_trace`] to record the same
    /// instruction-location sequence the tree-walker's `Inspector` reports.
    fn cur_ir_pc(&self, source: &ModuleSource) -> Option<super::IrPc> {
        let cm = source.get(self.module)?;
        let (block, inst) = cm.progs[self.cur].src.get(self.pc).copied().flatten()?;
        if inst & SRC_TERM != 0 {
            return None; // terminator — non-steppable (see `Program::src`)
        }
        Some(super::IrPc {
            module: self.module as u32,
            func: self.cur as FuncIdx,
            block: block as usize,
            inst: inst as usize,
        })
    }

    /// Run the continuation for at most `budget` ops, then return [`Outcome::Suspended`] at the next
    /// op boundary with the cursor persisted into `self` (resume by calling again); return
    /// [`Outcome::Done`] when the entry activation returns, or `Err` on a trap. Per-op fuel is
    /// charged here, one charge per op, exactly as the run-to-completion form did — slicing only
    /// chooses *where* to pause, never *what* runs, so the result is independent of `budget`.
    ///
    /// The cursor (`cur`/`base`/`pc`) lives in locals for the duration of the loop so the optimizer
    /// keeps it in registers; it is written back to `self` only when the loop exits (suspend), which
    /// is also what a future blocking-op / debug-stop seam will do before yielding.
    fn resume(
        &mut self,
        source: &ModuleSource,
        table: &SharedSlots,
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut HostCell,
        mut budget: u64,
    ) -> Result<Outcome, Trap> {
        let mut module = self.module;
        let mut cur = self.cur;
        let mut base = self.base;
        let mut pc = self.pc;
        // THREADS.md 4c-domain: the shared module source is read through a per-vCPU **lock-free local
        // cache** (`Arc` clones), refreshed only on a miss (a unit installed since the last sync). The
        // active module is held as an owned `Arc<Compiled>` (`c`) — independent of `local`, so a refresh
        // can't invalidate it — re-resolved only when an activation crosses modules (so the per-op hot
        // path, `c.*` via `Arc` deref, is unchanged). `resolve!` returns the `Arc` for a module index.
        let mut local: Vec<std::sync::Arc<Compiled>> = source.snapshot();
        macro_rules! resolve {
            ($m:expr) => {{
                let m = $m as usize;
                if m >= local.len() {
                    local = source.snapshot(); // miss: a module installed since last sync
                }
                match local.get(m) {
                    Some(a) => std::sync::Arc::clone(a),
                    None => return Err(Trap::Malformed), // forged/stale module index (defensive)
                }
            }};
        }
        let mut c: std::sync::Arc<Compiled> = resolve!(module);

        macro_rules! r {
            ($i:expr) => {
                self.regs[base + $i as usize]
            };
        }
        // Apply edge copies parallel-safely (a self-loop can alias src/dst): gather then scatter.
        macro_rules! edge {
            ($copies:expr) => {{
                let cp = $copies;
                if cp.aliasing {
                    // A destination is re-read as a source: gather all sources, then scatter, so a
                    // value isn't clobbered before it is read (a param swap/rotation).
                    self.scratch.clear();
                    for &(s, _) in cp.pairs.iter() {
                        self.scratch.push(self.regs[base + s as usize]);
                    }
                    for (k, &(_, d)) in cp.pairs.iter().enumerate() {
                        self.regs[base + d as usize] = self.scratch[k];
                    }
                } else {
                    // Non-aliasing (the common induction/accumulator edge): copy directly in one pass,
                    // no `scratch` traffic — sources and destinations are disjoint slot sets.
                    for &(s, d) in cp.pairs.iter() {
                        let v = self.regs[base + s as usize];
                        self.regs[base + d as usize] = v;
                    }
                }
            }};
        }
        // Fuel unification: fuel is metered at **IR safepoints** — a taken back-edge and each function
        // entry — not per op. A back-edge is a *backward* jump in the flat op array: blocks are laid
        // out in index order, so a terminator (the last op of its block, at index `pc`) taking a target
        // whose entry `$t <= pc` is exactly a branch to an earlier-or-same block. This bounds every
        // loop/recursion (an infinite loop must cross its back-edge unboundedly) while leaving
        // straight-line code free — the one unit the tree-walker and JIT can meter identically. The
        // per-op `budget` (suspension / single-step) is unchanged.
        macro_rules! backedge {
            ($t:expr) => {
                if ($t as usize) <= pc {
                    step(fuel, None)?;
                }
            };
        }

        loop {
            if budget == 0 {
                // Pause at this op boundary: persist the cursor so a later `resume` continues here.
                self.module = module;
                self.cur = cur;
                self.base = base;
                self.pc = pc;
                return Ok(Outcome::Suspended);
            }
            budget -= 1;
            match &c.progs[cur].ops[pc] {
                Op::Const { dst, val } => {
                    r!(*dst) = *val;
                    pc += 1;
                }
                Op::IntBin { dst, a, b, ty, op } => {
                    let v = match ty {
                        IntTy::I32 => Reg::from_i32(bin32(*op, r!(*a).i32(), r!(*b).i32())?),
                        IntTy::I64 => Reg::from_i64(bin64(*op, r!(*a).i64(), r!(*b).i64())?),
                    };
                    r!(*dst) = v;
                    pc += 1;
                }
                Op::IntCmp { dst, a, b, ty, op } => {
                    let res = match ty {
                        IntTy::I32 => cmp32(*op, r!(*a).i32(), r!(*b).i32()),
                        IntTy::I64 => cmp64(*op, r!(*a).i64(), r!(*b).i64()),
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::IntUn { dst, a, ty, op } => {
                    r!(*dst) = match ty {
                        IntTy::I32 => Reg::from_i32(intun32(*op, r!(*a).i32())),
                        IntTy::I64 => Reg::from_i64(intun64(*op, r!(*a).i64())),
                    };
                    pc += 1;
                }
                Op::Eqz { dst, a, ty } => {
                    let res = match ty {
                        IntTy::I32 => r!(*a).i32() == 0,
                        IntTy::I64 => r!(*a).i64() == 0,
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::Convert { dst, a, op } => {
                    r!(*dst) = match op {
                        ConvOp::ExtendI32S => Reg::from_i64(r!(*a).i32() as i64),
                        ConvOp::ExtendI32U => Reg::from_i64(r!(*a).i32() as u32 as i64),
                        ConvOp::WrapI64 => Reg::from_i32(r!(*a).i64() as i32),
                    };
                    pc += 1;
                }
                Op::Select { dst, cond, a, b } => {
                    r!(*dst) = if r!(*cond).i32() != 0 { r!(*a) } else { r!(*b) };
                    pc += 1;
                }
                Op::FBin { dst, a, b, ty, op } => {
                    r!(*dst) = match ty {
                        FloatTy::F32 => Reg::from_f32(fbin32(*op, r!(*a).f32(), r!(*b).f32())),
                        FloatTy::F64 => Reg::from_f64(fbin64(*op, r!(*a).f64(), r!(*b).f64())),
                    };
                    pc += 1;
                }
                Op::FUn { dst, a, ty, op } => {
                    r!(*dst) = match ty {
                        FloatTy::F32 => Reg::from_f32(fun32(*op, r!(*a).f32())),
                        FloatTy::F64 => Reg::from_f64(fun64(*op, r!(*a).f64())),
                    };
                    pc += 1;
                }
                Op::FCmp { dst, a, b, ty, op } => {
                    let res = match ty {
                        FloatTy::F32 => fcmp32(*op, r!(*a).f32(), r!(*b).f32()),
                        FloatTy::F64 => fcmp64(*op, r!(*a).f64(), r!(*b).f64()),
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::FToISat { dst, a, op } => {
                    r!(*dst) = fto_i(*op, r!(*a));
                    pc += 1;
                }
                Op::FToITrap { dst, a, op } => {
                    r!(*dst) = trunc_trap(*op, r!(*a))?;
                    pc += 1;
                }
                Op::IToFConv { dst, a, op } => {
                    r!(*dst) = i_to_f(*op, r!(*a));
                    pc += 1;
                }
                Op::Cast { dst, a, op } => {
                    r!(*dst) = cast(*op, r!(*a));
                    pc += 1;
                }
                Op::RefFunc { dst, func } => {
                    r!(*dst) = Reg::from_i32(*func as i32);
                    pc += 1;
                }
                Op::Load {
                    dst,
                    addr,
                    op,
                    offset,
                } => {
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let a = r!(*addr).i64() as u64;
                    r!(*dst) = m.load_scalar(a, *offset, *op)?;
                    pc += 1;
                }
                Op::Store {
                    addr,
                    value,
                    op,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let lo = r!(*value).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .store_scalar(a, *offset, *op, lo)?;
                    pc += 1;
                }
                // Bulk-memory ops (D62): both `MemCopy` and `MemMove` use the overlap-safe fast path
                // (bulk `memmove` on the backing behind the same whole-span confinement; the tree-walk
                // oracle keeps the scalar `mem_copy`).
                Op::MemCopy { dst, src, len } | Op::MemMove { dst, src, len } => {
                    let d = r!(*dst).i64() as u64;
                    let s = r!(*src).i64() as u64;
                    let n = r!(*len).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .mem_copy_fast(d, s, n)?;
                    pc += 1;
                }
                Op::MemFill { dst, val, len } => {
                    let d = r!(*dst).i64() as u64;
                    let v = r!(*val).i32() as u8;
                    let n = r!(*len).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .mem_fill_fast(d, v, n)?;
                    pc += 1;
                }
                Op::AtomicLoad {
                    dst,
                    addr,
                    ty,
                    offset,
                } => {
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let a = r!(*addr).i64() as u64;
                    r!(*dst) = Reg::from_value(m.atomic_load(a, *offset, *ty)?);
                    pc += 1;
                }
                Op::AtomicStore {
                    addr,
                    value,
                    ty,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let v = Value::I64(r!(*value).i64());
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_store(a, *offset, *ty, v)?;
                    pc += 1;
                }
                Op::AtomicRmw {
                    dst,
                    addr,
                    value,
                    ty,
                    op,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let v = Value::I64(r!(*value).i64());
                    let res = mem
                        .as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_rmw(a, *offset, *ty, *op, v)?;
                    r!(*dst) = Reg::from_value(res);
                    pc += 1;
                }
                Op::AtomicCmpxchg {
                    dst,
                    addr,
                    expected,
                    replacement,
                    ty,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let exp = Value::I64(r!(*expected).i64());
                    let rep = Value::I64(r!(*replacement).i64());
                    let res = mem
                        .as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_cmpxchg(a, *offset, *ty, exp, rep)?;
                    r!(*dst) = Reg::from_value(res);
                    pc += 1;
                }
                Op::Br { copies, target } => {
                    backedge!(*target);
                    edge!(copies);
                    pc = *target as usize;
                }
                Op::BrIf {
                    cond,
                    then_copies,
                    then_pc,
                    else_copies,
                    else_pc,
                } => {
                    if r!(*cond).i32() != 0 {
                        backedge!(*then_pc);
                        edge!(then_copies);
                        pc = *then_pc as usize;
                    } else {
                        backedge!(*else_pc);
                        edge!(else_copies);
                        pc = *else_pc as usize;
                    }
                }
                // Slice 5a fused compare+branch. Fuel is charged only if the taken edge is a
                // back-edge (fuel unification) — the fused-away `IntCmp` no longer needs its own
                // per-op charge, so fusion now saves a full op's work on the loop back-edge.
                Op::BrIfCmp {
                    a,
                    b,
                    ty,
                    op,
                    then_copies,
                    then_pc,
                    else_copies,
                    else_pc,
                } => {
                    let taken = match ty {
                        IntTy::I32 => cmp32(*op, r!(*a).i32(), r!(*b).i32()),
                        IntTy::I64 => cmp64(*op, r!(*a).i64(), r!(*b).i64()),
                    };
                    if taken {
                        backedge!(*then_pc);
                        edge!(then_copies);
                        pc = *then_pc as usize;
                    } else {
                        backedge!(*else_pc);
                        edge!(else_copies);
                        pc = *else_pc as usize;
                    }
                }
                Op::BrTable { idx, arms, default } => {
                    let i = r!(*idx).i32() as u32 as usize;
                    let (copies, target) = arms.get(i).unwrap_or(default);
                    backedge!(*target);
                    edge!(copies);
                    pc = *target as usize;
                }
                // `<setjmp.h>` `setjmp`: checkpoint the resume point (the op after this, in this
                // activation) keyed by the guest `jmp_buf` address, and return 0. The register window
                // survives in place (per-block slots are distinct), so no snapshot is taken.
                Op::SetJmp { buf, dst } => {
                    let key = r!(*buf).i64() as u64;
                    self.setjmp_points.insert(
                        key,
                        ByteSetJmp {
                            depth: self.stack.len(),
                            module,
                            cur,
                            base,
                            pc: pc + 1,
                            dst: *dst,
                        },
                    );
                    r!(*dst) = Reg::from_i32(0);
                    pc += 1;
                }
                // `<setjmp.h>` `longjmp`: pop the activation stack back to the checkpoint (intervening
                // activations discarded — C has no cleanups), restore its cursor, and re-enter with the
                // `setjmp` result set to `val` (a `0` becomes `1`, per C). A missing checkpoint or one
                // whose activation already returned traps in-sandbox (§3b totality).
                Op::LongJmp { buf, val } => {
                    let key = r!(*buf).i64() as u64;
                    let v = r!(*val).i32();
                    let resume = if v == 0 { 1 } else { v };
                    let point = *self.setjmp_points.get(&key).ok_or(Trap::Malformed)?;
                    if point.depth > self.stack.len() {
                        return Err(Trap::Malformed); // the setjmp activation already returned
                    }
                    self.stack.truncate(point.depth);
                    module = point.module;
                    cur = point.cur;
                    base = point.base;
                    pc = point.pc;
                    c = resolve!(module);
                    self.regs[base + point.dst as usize] = Reg::from_i32(resume);
                }
                Op::Call { callee, args, dst } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let callee = *callee as usize;
                    #[cfg(feature = "callprof")]
                    if module == 0 {
                        callprof::hit(callee);
                    }
                    // wasm-JIT tier-up: a module-0 direct call to an eligible function surfaces to the
                    // host, which runs the emitted region and delivers the results. `argv` is the raw
                    // i64 arg slots; the host reads them per the callee's signature. Suspension-free by
                    // construction (`mixed_ok`), so this is a plain "fast call": spill past the op and
                    // resume with the results in `dst` (`deliver_tierup`), exactly like an interp call.
                    if module == 0
                        && self
                            .jit_eligible
                            .as_ref()
                            .is_some_and(|e| e.get(callee).copied().unwrap_or(false))
                    {
                        // #717 host sync: snapshot the window's scalar committed extent for the host
                        // to write into the emitted `"mapped"` global. A window whose page state is
                        // not representable by one bound (sparse grow, `Ro`/`Unmapped`/aliased pages)
                        // declines tier-up — fall through to the interpreted call below, which honors
                        // the full per-page map (fail-closed; the interpreter is always right). A
                        // memory-less module has nothing to bound (no emitted access): sync `0`.
                        //
                        // #750 paged tier: the emitted code carries a per-access page check, so an
                        // unrepresentable window must NOT decline — surface with the reserved size
                        // (the bound must never under-admit a page the driver's table admits).
                        let extent = match mem.as_ref() {
                            None => Some(0),
                            Some(m) if self.jit_page_checked => Some(m.reserved_size()),
                            Some(m) => m.scalar_extent(),
                        };
                        if let Some(mapped) = extent {
                            let argv: Box<[i64]> = args.iter().map(|a| r!(*a).i64()).collect();
                            let results: Box<[ValType]> = c.result_types[callee].clone().into();
                            // Spill past the call with the caller's window intact (no callee frame
                            // pushed); `deliver_tierup` writes the results into `dst` relative to
                            // this base.
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::TierUp {
                                func: callee as u32,
                                argv,
                                dst: *dst as usize,
                                results,
                                mapped,
                            });
                        }
                    }
                    // A direct call stays in the current module.
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack
                        .push((module, cur, base, pc + 1, base + *dst as usize));
                    cur = callee;
                    base = nb;
                    pc = 0;
                }
                Op::CallIndirect {
                    idx,
                    args,
                    dst,
                    want_params,
                    want_results,
                } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                                       // Resolve through the **runtime dispatch table** (slot ⇒ (module, func)); an empty
                                       // padding slot or a signature mismatch is an inert IndirectCallType trap. The
                                       // target may be an installed §22 unit (a different module) — a cross-module call.
                    let slot = (r!(*idx).i32() as u32 as usize) & (table.len() - 1);
                    let ts = table.slot(slot);
                    if ts.module == super::TABLE_EMPTY {
                        return Err(Trap::IndirectCallType);
                    }
                    let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
                    let tm = resolve!(tmod);
                    let (cp, cr) = &tm.sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + tm.progs[tfunc].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack
                        .push((module, cur, base, pc + 1, base + *dst as usize));
                    if tmod != module {
                        module = tmod;
                        c = tm;
                    }
                    cur = tfunc;
                    base = nb;
                    pc = 0;
                }
                Op::Ret { srcs } => match self.stack.pop() {
                    None => {
                        let tys = &c.result_types[cur];
                        return Ok(Outcome::Done(
                            srcs.iter()
                                .zip(tys)
                                .map(|(s, ty)| self.regs[base + *s as usize].to_value(*ty))
                                .collect(),
                        ));
                    }
                    Some((cmod, cprog, cbase, cpc, ret_abs)) => {
                        for (i, s) in srcs.iter().enumerate() {
                            self.regs[ret_abs + i] = self.regs[base + *s as usize];
                        }
                        if cmod != module {
                            module = cmod;
                            c = resolve!(cmod);
                        }
                        cur = cprog;
                        base = cbase;
                        pc = cpc;
                    }
                },
                // Tail calls reuse the *current* window (`base` unchanged) instead of pushing a
                // return entry, so the callee returns to this activation's caller. Args may alias the
                // destination prefix, so gather into `scratch` then scatter (like edge copies).
                Op::TailCall { callee, args } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let callee = *callee as usize;
                    #[cfg(feature = "callprof")]
                    if module == 0 {
                        callprof::hit(callee);
                    }
                    let need = base + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    self.scratch.clear();
                    for a in args.iter() {
                        self.scratch.push(self.regs[base + *a as usize]);
                    }
                    for (i, &v) in self.scratch.iter().enumerate() {
                        self.regs[base + i] = v;
                    }
                    cur = callee;
                    pc = 0;
                }
                Op::TailCallIndirect {
                    idx,
                    args,
                    want_params,
                    want_results,
                } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let slot = (r!(*idx).i32() as u32 as usize) & (table.len() - 1);
                    let ts = table.slot(slot);
                    if ts.module == super::TABLE_EMPTY {
                        return Err(Trap::IndirectCallType);
                    }
                    let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
                    let tm = resolve!(tmod);
                    let (cp, cr) = &tm.sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let need = base + tm.progs[tfunc].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    self.scratch.clear();
                    for a in args.iter() {
                        self.scratch.push(self.regs[base + *a as usize]);
                    }
                    for (i, &v) in self.scratch.iter().enumerate() {
                        self.regs[base + i] = v;
                    }
                    if tmod != module {
                        module = tmod;
                        c = tm;
                    }
                    cur = tfunc;
                    pc = 0;
                }
                Op::CapCall {
                    type_id,
                    op,
                    handle,
                    params,
                    args,
                    dst,
                    results,
                } => {
                    // Generic synchronous powerbox dispatch — the same path and ABI the tree-walker's
                    // generic `CapCall` arm uses (`cap_dispatch_slots`): handle as an i32, args/results
                    // as i64 slots, results re-typed by the call's `sig.results`. Via [`HostCell`] so a
                    // parallel vCPU takes the shared-host lock only for this one call (4c-host); the
                    // cooperative path is exclusive (uncontended), so order is unchanged.
                    // `u32::MAX` = no handle operand (a v8 `call.import` — the slot binding
                    // identifies the capability; the dispatch ignores the value).
                    let h = if *handle == u32::MAX {
                        0
                    } else {
                        r!(*handle).i32()
                    };
                    let mut argv: Vec<i64> = Vec::with_capacity(args.len());
                    for a in args.iter() {
                        argv.push(r!(*a).i64());
                    }
                    // §22: an **import-bound** `Jit` driver op (`invoke`/`install`/`uninstall`) can't be
                    // serviced by the generic `cap_dispatch_slots` — it needs the scheduler-owning
                    // driver, exactly like a *static* `cap.call (JIT, op)` (which lowers straight to
                    // `Op::JitInvoke`). Resolve the binding and surface the same driver `Outcome`, so a
                    // self-hosted guest whose `__vm_jit_*` are lowered to `call.import` (svm-llvm) drives
                    // the Jit cap on the bytecode engine too (the pure-host `compile`/`compile_linked`
                    // ops 0/5 stay on `cap_dispatch_slots` below). The submitted unit is re-verified by
                    // the embedder's `jit_validator` before it runs — the security hinge is unchanged.
                    if *type_id == svm_ir::CAP_IMPORT_TYPE_ID {
                        if let Some(b) = host.with(|p| p.import_binding(*op)) {
                            if b.bound
                                && b.type_id == super::cap_id::JIT
                                && matches!(b.op, 1 | 3 | 4)
                            {
                                if argv.is_empty() {
                                    return Err(Trap::CapFault); // invoke/install/uninstall need arg0
                                }
                                self.module = module;
                                self.cur = cur;
                                self.base = base;
                                self.pc = pc + 1;
                                return Ok(match b.op {
                                    3 => Outcome::JitInstall {
                                        h: b.handle,
                                        code: argv[0] as i32,
                                        dst: *dst,
                                    },
                                    4 => Outcome::JitUninstall {
                                        h: b.handle,
                                        slot: argv[0],
                                        dst: *dst,
                                    },
                                    _ => Outcome::JitInvoke {
                                        h: b.handle,
                                        code: argv[0] as i32,
                                        argv: argv[1..].to_vec().into_boxed_slice(),
                                        dst: *dst,
                                        // The unit entry's params are the call's params minus arg0 (code).
                                        params: params
                                            .get(1..)
                                            .unwrap_or(&[])
                                            .to_vec()
                                            .into_boxed_slice(),
                                        results: results.clone(),
                                    },
                                });
                            }
                        }
                    }
                    // §3.6 (I36 slice 2) — caller-side parking: a call through a live-callee
                    // offer never reaches the generic dispatch. It enqueues on the callee's
                    // inbound queue and parks this task until the handler's reply (the
                    // tree-walker's caller-parking arm, task-level). A full callee queue is
                    // probeable backpressure (`EAGAIN` as the call's result), never a trap.
                    if let Some((callee, export)) = host.with(|p| p.live_impl_of(h, *type_id)) {
                        let t = callee.lock_unpoisoned().svc_enqueue(export, *op, argv);
                        if let Some(ticket) = t {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::LiveCall {
                                ticket,
                                callee,
                                dst: *dst,
                            });
                        }
                        if !results.is_empty() {
                            self.regs[base + *dst as usize] = Reg::from_i64(super::EAGAIN);
                        }
                        pc += 1;
                        continue;
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut pending_id = None;
                    let res = host.with(|p| {
                        p.cap_dispatch_slots_pending(*type_id, *op, h, &argv, gm, &mut pending_id)
                    })?;
                    // §12 parking-on-blocking: a punted offloadable dispatch. The `with` scope
                    // above already released the shared-host lock. The exactly-`i64` case is
                    // surfaced as [`Outcome::CapPending`] so the DRIVER chooses the wait shape
                    // (F2: the cooperative `drive` fiber-parks a punting fiber; every other
                    // driver waits inline — the I45 posture). Other reply shapes keep the
                    // slice-1 inline wait right here; the placeholder `res` is discarded.
                    if let Some(id) = pending_id {
                        if results.len() > 1 {
                            // Parkable ops carry a single-slot scalar reply (invariant 8) —
                            // a wider declared signature is a registration bug, fail-closed
                            // BEFORE any park or wait.
                            return Err(Trap::CapFault);
                        }
                        if let [ValType::I64] = &results[..] {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::CapPending { id, dst: *dst });
                        }
                        let comps = host.with(|p| p.completions());
                        let r = comps.wait(id);
                        if let Some(ty) = results.first() {
                            self.regs[base + *dst as usize] = Reg::from_value(slot_to_val(*ty, r));
                        }
                        pc += 1;
                        continue;
                    }
                    // Blocking-stdin park: a `Stream{In}` `read` (type 0, op 0) whose buffer was empty
                    // under `Host::set_stdin_blocking` yields here instead of completing. Do NOT write
                    // results or advance `pc`: persist state at *this* instruction so the driver, after
                    // pushing more input, re-issues the read on resume. Gated on the stream-read op so
                    // no other cap.call pays the flag check.
                    // A `call.import` dispatch carries the `CAP_IMPORT_TYPE_ID` sentinel — read
                    // its bound `(type_id, op)` so an imported stdin `read` parks exactly like the
                    // resolved `cap.call` form (IMPORTS.md phase 3).
                    let (eff_tid, eff_op) = if *type_id == svm_ir::CAP_IMPORT_TYPE_ID {
                        host.with(|p| p.import_binding(*op))
                            .map(|b| (b.type_id, b.op))
                            .unwrap_or((*type_id, *op))
                    } else {
                        (*type_id, *op)
                    };
                    if eff_tid == super::cap_id::STREAM
                        && eff_op == 0
                        && host.with(|p| p.take_stdin_parked())
                    {
                        self.module = module;
                        self.cur = cur;
                        self.base = base;
                        self.pc = pc;
                        return Ok(Outcome::StdinPark);
                    }
                    for (i, (s, ty)) in res.iter().zip(results.iter()).enumerate() {
                        self.regs[base + *dst as usize + i] = Reg::from_value(slot_to_val(*ty, *s));
                    }
                    pc += 1;
                }
                Op::SvcPoll { dst, wait } => {
                    // §3.6 serve-loop core (I36 slice 1), the tree-walk serve arm's rewind state
                    // machine in register-window form. A handler that just returned re-entered
                    // this op via its rewound linkage with its result in `dst` — settle it into
                    // the ticket's completion cell. No cross-domain caller can be parked on the
                    // ticket in this engine yet (caller-side parking is a later I36 slice), so
                    // the reply always rides the cell — the tree-walker's unclaimed-result path.
                    if let Some(t) = self.serve_ticket.take() {
                        let v = self.regs[base + *dst as usize].i64();
                        host.with(|p| p.svc_results.insert(t, v));
                        self.serve_count += 1;
                    }
                    // Admit queued dispatches: un-servable ones settle inline with a probeable
                    // errno (the dispatch's fault, never the domain's — it keeps serving); the
                    // first servable one switches into a handler activation whose return linkage
                    // re-executes this op (pc deliberately NOT advanced).
                    let mut admitted = false;
                    loop {
                        let d = host.with(|p| p.svc_queue.pop_front());
                        let Some(d) = d else { break };
                        // The queue only holds servable dispatches (checked at enqueue), so a
                        // missing handler here is host-state corruption: fail closed. Handlers
                        // are the domain's home-module functions (`self.home` — the primary, or
                        // a separate-module child's own unit); serving from any other unit
                        // would resolve indices against the wrong program table.
                        let fidx = host
                            .with(|p| p.svc_handler_func(d.export, d.op))
                            .ok_or(Trap::CapFault)? as usize;
                        if module != self.home {
                            return Err(Trap::CapFault);
                        }
                        let (params, _) = c.sigs.get(fidx).ok_or(Trap::CapFault)?;
                        if d.args.len() != params.len() {
                            host.with(|p| p.svc_results.insert(d.ticket, super::EINVAL));
                            continue;
                        }
                        let nb = base + c.progs[cur].nslots as usize;
                        let need = nb + c.progs[fidx].nslots as usize;
                        if self.regs.len() < need {
                            self.regs.resize(need, Reg::default());
                        }
                        for (i, (s, ty)) in d.args.iter().zip(params.iter()).enumerate() {
                            self.regs[nb + i] = Reg::from_value(slot_to_val(*ty, *s));
                        }
                        self.stack
                            .push((module, cur, base, pc, base + *dst as usize));
                        self.serve_ticket = Some(d.ticket);
                        cur = fidx;
                        base = nb;
                        pc = 0;
                        admitted = true;
                        break;
                    }
                    if !admitted {
                        if *wait && self.serve_count == 0 {
                            // svc.wait with no progress: persist the cursor AT this op (a wake
                            // re-executes the whole drain) and park the task on its domain.
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc;
                            return Ok(Outcome::SvcWait);
                        }
                        // Queue drained: deliver the completed count and close the activation.
                        self.regs[base + *dst as usize] = Reg::from_i64(self.serve_count);
                        self.serve_count = 0;
                        pc += 1;
                    }
                }
                Op::CapSelfExt { op, handle, dst } => {
                    // §3.5 self-namespace extensions — through the shared &mut dispatch entry
                    // (interning / reification mutate host state), same as the tree-walker.
                    let argv: Vec<i64> = match handle {
                        Some(h) => vec![r!(*h).i32() as i64],
                        None => Vec::new(),
                    };
                    let res = host.with(|p| {
                        p.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, *op, 0, &argv, None)
                    })?;
                    r!(*dst) = Reg::from_i32(*res.first().ok_or(Trap::CapFault)? as i32);
                    pc += 1;
                }
                // §12 fiber ops escape to `drive` (which owns the registry / resume chain). Each
                // advances past itself and persists the cursor, so the driver — after creating the
                // fiber, switching in, or switching back — resumes this activation right after the op
                // (with the op's `dst` slot(s) filled in by the driver).
                Op::ContNew { func, sp, dst } => {
                    let funcref = r!(*func).i32();
                    let spv = r!(*sp).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContNew {
                        funcref,
                        sp: spv,
                        dst,
                    });
                }
                Op::ContResume {
                    k,
                    arg,
                    dst,
                    blocking,
                } => {
                    // Fuel unification: charge one fuel per `cont.resume` op — the tree-walker charges
                    // the same at its `Inst::ContResume` arm. Resuming a fiber is a control transfer
                    // per-op fuel used to meter; without this, a long fiber-resume chain runs unmetered.
                    step(fuel, None)?;
                    let kh = r!(*k).i32();
                    let arg = r!(*arg).i64();
                    let dst = *dst;
                    let blocking = *blocking;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    // I48: a blocking park rewinds the resumer's cursor to THIS op (via `resume_ip`)
                    // so the wake re-executes it; `pc` here is this op's index (the cursor is written
                    // back as `pc + 1` for the ordinary switch/poll continuation).
                    let resume_ip = pc;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContResume {
                        kh,
                        arg,
                        dst,
                        blocking,
                        resume_ip,
                    });
                }
                Op::Suspend { value, dst } => {
                    let value = r!(*value).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::FiberSuspend { value, dst });
                }
                // §12 multi-vCPU ops escape to the `drive` scheduler (which owns the task set). Each
                // advances past itself and persists the cursor, so the scheduler resumes this
                // activation right after the op with the op's `dst` filled in.
                Op::ThreadSpawn { func, sp, arg, dst } => {
                    let sp = r!(*sp).i64();
                    let arg = r!(*arg).i64();
                    let (func, dst) = (*func, *dst);
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    // `module` is the executing frame's module: an installed unit's spawn resolves
                    // `func` in the unit, not module 0 (the verifier checked it there).
                    return Ok(Outcome::ThreadSpawn {
                        func,
                        sp,
                        arg,
                        dst,
                        module,
                    });
                }
                Op::ThreadJoin { handle, dst } => {
                    let handle = r!(*handle).i32();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ThreadJoin { handle, dst });
                }
                // §14 executor children — the Instantiator authority `(ibase, isize)` is resolved here
                // (a forged/ungranted cap is an inert CapFault in place), then the driver builds the
                // confined child (it owns the task set + the per-child environments).
                Op::ChildOffer {
                    handle,
                    child,
                    export,
                    dst,
                } => {
                    // The family-level authority check (as the tree-walker's Instantiator arm):
                    // a forged/wrong-type handle is a CapFault before the op logic runs.
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?;
                    let child = r!(*child).i32();
                    let export = r!(*export).i64() as u32;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ChildOffer { child, export, dst });
                }
                Op::CloneCaller {
                    args,
                    dst,
                    has_result,
                } => {
                    // Arity picks the mode (mirrors the oracle, `svm-interp` clone_caller arm):
                    // 2 args = explicit `(reply_orig, reply_twin)`; 0/1 args = pid mode
                    // (`reply_orig = None` → the parent gets the twin's task id).
                    let (reply_orig, reply_twin) = if args.len() >= 2 {
                        (Some(r!(args[0]).i64()), r!(args[1]).i64())
                    } else {
                        let twin = args.first().map(|a| r!(*a).i64()).unwrap_or(0);
                        (None, twin)
                    };
                    let dst = *dst;
                    let has_result = *has_result;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::CloneCaller {
                        reply_orig,
                        reply_twin,
                        dst,
                        has_result,
                    });
                }
                Op::Reap {
                    pid,
                    dst,
                    has_result,
                } => {
                    // The pid to reap (an out-of-range default → the driver answers -ECHILD).
                    let pid = pid.map(|p| r!(p).i64()).unwrap_or(-1);
                    let dst = *dst;
                    let has_result = *has_result;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Reap {
                        pid,
                        dst,
                        has_result,
                    });
                }
                Op::Instantiate {
                    handle,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
                    // op 11: resolve the grant-list `(ptr, count)` from their registers (op 0 is None).
                    let grants = grants.map(|(pr, nr)| (r!(pr).i64() as u64, r!(nr).i64() as u64));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Instantiate {
                        ibase,
                        isize: isz,
                        entry,
                        off,
                        size_log2,
                        quota,
                        dst,
                        grants,
                        budget: 0,
                    });
                }
                // §14 separate-module executor child — like `Instantiate`, but the first arg is a
                // granted `Module` handle (the slot ABI crosses it as an i64; low 32 bits) whose
                // program the driver resolves + compiles + runs.
                Op::InstantiateModule {
                    handle,
                    module: module_reg,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let mh = r!(*module_reg).i64() as i32;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
                    // op 13: resolve the grant-list `(ptr, count)` from their registers (op 5 is None).
                    let grants = grants.map(|(pr, nr)| (r!(pr).i64() as u64, r!(nr).i64() as u64));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::InstantiateModule {
                        ibase,
                        isize: isz,
                        mh,
                        entry,
                        off,
                        size_log2,
                        quota,
                        dst,
                        grants,
                        budget: 0,
                    });
                }
                // CONSOLIDATION.md §3d — `instantiate_rec` (op 17): read the 56-byte record from
                // this vCPU's confined window and fail closed exactly as the tree-walker's arm does
                // (bad version / pager / budget-quota mix / dangling budget handle → `CapFault`).
                // This tier never natively demand-pages: the module-level entries decline any
                // op-17 module with impl exports ([`compile_module_for`]), so a surviving pager
                // field can only come from an export-less module — which `CapFault`s identically
                // on the tree-walker. Geometry validation and the budget **drain** stay at the
                // drivers' commit sites (`event_instantiate*` / the drive arms) so a refused spawn
                // leaves the budget intact — the tree-walker's peek-then-drain discipline. (One
                // known error-order seam, shared with ops 11/13: an invalid grant list *plus* bad
                // geometry lands `-EINVAL` here but `CapFault` on the tree-walker, because grant
                // records are parsed at the drivers' construction step.)
                Op::InstantiateRec { handle, rec, dst } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let rp = r!(*rec).i64() as u64;
                    let raw = mem.as_ref().ok_or(Trap::Malformed)?.read_window(rp, 56)?;
                    let raw: &[u8; 56] = raw.as_slice().try_into().map_err(|_| Trap::Malformed)?;
                    // Shared 56-byte layout decode (#911); pager/budget handling stays tier-local.
                    let sr = SpawnRec::parse(raw).ok_or(Trap::CapFault)?; // version — fail closed
                    let entry = sr.entry as i64;
                    let off = sr.off as i64;
                    let size_log2 = sr.size_log2;
                    let modh = sr.modh;
                    let budget = sr.budget;
                    let quota = sr.quota;
                    if sr.pager != u32::MAX {
                        return Err(Trap::CapFault); // no impl exports here (see above) — fail closed
                    }
                    if budget != 0 {
                        if quota != 0 {
                            return Err(Trap::CapFault); // budget + raw quota is ambiguous
                        }
                        // Validate the handle now (dangling → CapFault, as the tree-walker);
                        // the drain waits for the drivers' commit.
                        host.with(|p| p.peek_budget(budget).map(|_| ()).ok_or(Trap::CapFault))?;
                    }
                    let grants = (sr.grants_n > 0).then_some((sr.grants_ptr, sr.grants_n));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(if modh >= 0 {
                        Outcome::InstantiateModule {
                            ibase,
                            isize: isz,
                            mh: modh,
                            entry,
                            off,
                            size_log2,
                            quota,
                            dst,
                            grants,
                            budget,
                        }
                    } else {
                        Outcome::Instantiate {
                            ibase,
                            isize: isz,
                            entry,
                            off,
                            size_log2,
                            quota,
                            dst,
                            grants,
                            budget,
                        }
                    });
                }
                // §14 `join` — check the Instantiator authority, then reuse the thread join machinery
                // (executor children live in the same `threads` handle namespace as `thread.spawn`).
                Op::InstJoin { handle, child, dst } => {
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?; // authority
                    let handle = r!(*child).i32();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ThreadJoin { handle, dst });
                }
                Op::MemoryWait {
                    ty,
                    addr,
                    expected,
                    timeout,
                    dst,
                } => {
                    // Validate the address (confine/align/prot — traps surface here), mirroring
                    // `Inst::MemoryWait`; the scheduler does the value compare + park/wake.
                    let width = super::atomic_width(*ty);
                    let a = r!(*addr).i64() as u64;
                    let expected = r!(*expected).lo & super::width_mask(width);
                    let to_ns = r!(*timeout).i64();
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base_addr = m.prepare_wait(a, *ty)?;
                    let max = super::MAX_WAIT.as_nanos() as u64;
                    let timeout = if to_ns < 0 {
                        max
                    } else {
                        (to_ns as u64).min(max)
                    };
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::MemoryWait {
                        base: base_addr,
                        expected,
                        width,
                        timeout,
                        dst,
                    });
                }
                Op::MemoryNotify { addr, count, dst } => {
                    let a = r!(*addr).i64() as u64;
                    let count = r!(*count).i32();
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base_addr = m.confine_for_notify(a)?;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::MemoryNotify {
                        base: base_addr,
                        count,
                        dst,
                    });
                }
                // §22 install/uninstall escape to the driver, which owns the (mutable) dispatch table
                // and module set. Authority is resolved there (a forged handle is an inert CapFault).
                Op::JitInstall { handle, code, dst } => {
                    let h = r!(*handle).i32();
                    let code = r!(*code).i64() as i32;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitInstall { h, code, dst });
                }
                Op::JitUninstall { handle, slot, dst } => {
                    let h = r!(*handle).i32();
                    let slot = r!(*slot).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitUninstall { h, slot, dst });
                }
                Op::JitInvoke {
                    handle,
                    code,
                    args,
                    dst,
                    params,
                    results,
                } => {
                    let h = r!(*handle).i32();
                    let code = r!(*code).i64() as i32;
                    let argv: Box<[i64]> = args.iter().map(|a| r!(*a).i64()).collect();
                    // `params`/`results` live in this op (in `mods`), which the driver may reallocate
                    // when it pushes the invoked unit — so hand owned copies up.
                    let (dst, params, results) = (*dst, params.clone(), results.clone());
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitInvoke {
                        h,
                        code,
                        argv,
                        dst,
                        params,
                        results,
                    });
                }
                Op::GcRoots {
                    lo,
                    hi,
                    mask,
                    buf,
                    cap,
                    dst,
                } => {
                    let lo = r!(*lo).i64() as u64;
                    let hi = r!(*hi).i64() as u64;
                    let mask = r!(*mask).i64() as u64;
                    // Security (GC.md §3/§6): the payload mask may only clear the top byte, else a host
                    // word could be folded into the guest window past the range filter. (The verifier
                    // rejects a constant fold-down mask; this defends an unverified / non-constant mask.)
                    if mask | 0xFF00_0000_0000_0000 != u64::MAX {
                        return Err(Trap::Malformed);
                    }
                    let buf = r!(*buf).i64() as u64;
                    let cap = r!(*cap).i64().max(0) as usize;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::GcRoots {
                        lo,
                        hi,
                        mask,
                        buf,
                        cap,
                        dst,
                    });
                }
                Op::Unreachable => return Err(Trap::Unreachable),
                Op::Eval {
                    inst,
                    block_base,
                    dst,
                } => {
                    // Run the op against this block's sub-window with its original block-local operand
                    // indices; reuse the reference semantics. `eval_inst` borrows the window immutably
                    // and `mem` mutably (disjoint), so we read the result before writing it back.
                    let win_lo = base + *block_base as usize;
                    let win_hi = base + c.progs[cur].nslots as usize;
                    let r = super::eval_inst(inst, &self.regs[win_lo..win_hi], mem)?;
                    if let Some(v) = r {
                        self.regs[base + *dst as usize] = v;
                    }
                    pc += 1;
                }
                Op::DurableShadowBase { dst } => {
                    // §12.8 4A.5: this context's shadow-SP word address (its own region base).
                    self.regs[base + *dst as usize] =
                        Reg::from_i64(self.durable_region_base as i64);
                    pc += 1;
                }
                Op::VcpuTlsGet { dst } => {
                    // §12 per-vCPU TLS read: this vCPU's word (seeded to its dense id, guest-overwritable).
                    self.regs[base + *dst as usize] = Reg::from_i64(self.tls);
                    pc += 1;
                }
                Op::VcpuTlsSet { val } => {
                    self.tls = r!(*val).i64();
                    pc += 1;
                }
            }
        }
    }
}
