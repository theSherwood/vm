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
    IToF, Inst, IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValType, VarLoc,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, slot_to_val, step, trunc_trap, val_to_slot, GuestMem, Host, Mem, Reg, Trap,
    Value, VarValue, DEFAULT_RESERVED_LOG2,
};

/// Block-argument moves applied on a taken edge: `(src_slot, dst_slot)` pairs (frame-relative).
type Copies = Box<[(u32, u32)]>;
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
    PtrAdd {
        dst: u32,
        a: u32,
        b: u32,
    },
    PtrCast {
        dst: u32,
        a: u32,
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
    /// §7 reflection `cap.self.count` — number of caps this domain holds (one `i32` result).
    CapSelfCount {
        dst: u32,
    },
    /// §6 attestation `cap.self.attest` — the domain's packed provenance (one `i32` result).
    CapSelfAttest {
        dst: u32,
    },
    /// §3.5 self-namespace extensions through the shared dispatch entry: `op` packs
    /// `(selfop | idx << 8)` (6 = `cap.self.type_id`, 7 = `cap.self.covers`, 8 =
    /// `export.handle`); `handle` is the optional live handle-register (covers only). One
    /// `i32` result.
    CapSelfExt {
        op: u32,
        handle: Option<u32>,
        dst: u32,
    },
    /// §7 reflection `cap.self.get` — the `idx`-th held cap as `(handle, type_id)` (two `i32`
    /// results in `dst`, `dst+1`).
    CapSelfGet {
        idx: u32,
        dst: u32,
    },
    /// §7 reflection `cap.self.resolve` — resolve a name buffer `(name_ptr, name_len)` to its handle
    /// (one `i32` result, `-errno` on miss). Routed through `cap_dispatch_slots` (op 2) like a cap.call.
    CapSelfResolve {
        name_ptr: u32,
        name_len: u32,
        dst: u32,
    },
    /// §7 reflection `cap.self.label` — write the handle's label into the window `(handle, buf_ptr,
    /// buf_cap)`, returning its length (one `i32`). Routed through `cap_dispatch_slots` (op 3).
    CapSelfLabel {
        handle: u32,
        buf_ptr: u32,
        buf_cap: u32,
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
    ContResume {
        k: u32,
        arg: u32,
        dst: u32,
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
    /// §14 `Instantiator.spawn_coroutine(entry, off, size_log2, fuel)` (op 2): spawn a cooperative
    /// coroutine child confined to `[off, off+2^size_log2)` of the holder's range, with a Yielder-only
    /// powerbox; its handle (or `EINVAL`) lands at `dst`. `handle` is the Instantiator cap (authority).
    SpawnCoroutine {
        handle: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        dst: u32,
        /// op 4 `spawn_demand_coroutine`: the child window starts unmapped (fault-driven yield).
        demand: bool,
    },
    /// §14 `Instantiator.spawn_coroutine_module(module, entry, off, size_log2, fuel)` (op 6): like
    /// [`Op::SpawnCoroutine`], but the cooperative child runs a host-granted **separate** `Module`
    /// (`module` is its handle, the first i64 arg). The driver resolves + compiles the module and
    /// materializes its data into the carve; thereafter it is `resume`d inline like any coroutine.
    /// `demand` selects op 7 `spawn_demand_coroutine_module` (data segments supplied lazily).
    SpawnCoroutineModule {
        handle: u32,
        module: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        dst: u32,
        demand: bool,
    },
    /// §14 `Instantiator.resume(ch, value)` (op 3): drive coroutine `ch` inline until it yields or
    /// returns; `(status, value)` land at `dst`/`dst+1`. `handle` is the Instantiator cap.
    CoResume {
        handle: u32,
        ch: u32,
        value: u32,
        dst: u32,
    },
    /// §14 `Yielder.yield(value)` (op 0): suspend this coroutine, hand `value` to the resumer; the
    /// next resume's value lands at `dst`. `handle` is the Yielder cap (authority).
    CoYield {
        handle: u32,
        value: u32,
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
        self.mods.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The primary program (module 0).
    fn primary(&self) -> std::sync::Arc<Compiled> {
        std::sync::Arc::clone(&self.mods.lock().unwrap_or_else(|e| e.into_inner())[0])
    }

    /// Module `i` (`0` = primary, `k≥1` = an installed unit), or `None` if out of range.
    fn get(&self, i: usize) -> Option<std::sync::Arc<Compiled>> {
        self.mods
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(i)
            .cloned()
    }

    /// Append a module (a §14 `instantiate_module` child's program) and return its index. (§22
    /// `Jit.install` instead goes through [`Domain::install`], which also fills a dispatch slot.)
    fn push(&self, unit: Compiled) -> usize {
        let mut mods = self.mods.lock().unwrap_or_else(|e| e.into_inner());
        mods.push(std::sync::Arc::new(unit));
        mods.len() - 1
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
    /// appended). `&self` (interior-mutable) so a shared `&Domain` can install. The whole op serializes
    /// under the source lock, and the slot store is `Release`, so a reader that observes the slot also
    /// observes the pushed unit.
    fn install(&self, unit: Compiled) -> Option<usize> {
        use std::sync::atomic::Ordering;
        let mut mods = self.source.mods.lock().unwrap_or_else(|e| e.into_inner());
        let slot = self
            .table
            .slots
            .iter()
            .position(|s| (s.load(Ordering::Relaxed) >> 32) as u32 == super::TABLE_EMPTY)?;
        mods.push(std::sync::Arc::new(unit));
        let module = (mods.len() - 1) as u32;
        self.table.slots[slot].store(super::pack_slot(module, 0), Ordering::Release);
        Some(slot)
    }

    /// `Jit.uninstall`: clear a filled padding slot (`≥ n_real`) back to trapping, returning success.
    /// A real-function slot (`< n_real`), out-of-range, or already-empty slot is rejected. The unit
    /// stays in `source` (append-only); only the slot is reclaimed. Serialized under the source lock.
    fn uninstall(&self, slot: usize, n_real: usize) -> bool {
        use std::sync::atomic::Ordering;
        let _g = self.source.mods.lock().unwrap_or_else(|e| e.into_inner());
        if slot >= n_real
            && slot < self.table.slots.len()
            && (self.table.slots[slot].load(Ordering::Relaxed) >> 32) as u32 != super::TABLE_EMPTY
        {
            self.table.slots[slot]
                .store(super::pack_slot(super::TABLE_EMPTY, 0), Ordering::Release);
            true
        } else {
            false
        }
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
}

fn scan_seams(funcs: &[Func]) -> Seams {
    let mut s = Seams::default();
    for f in funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    // ops 0/1 = instantiate/join, op 5 = instantiate_module (executor children);
                    // everything else on INSTANTIATOR/YIELDER is the inline coroutine round-trip.
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR,
                        op: 0 | 1 | 5 | 14,
                        ..
                    } => s.has_instantiate = true,
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR | super::cap_id::YIELDER,
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
                    Inst::ContNew { .. } | Inst::ContResume { .. } | Inst::Suspend { .. } => {
                        s.has_fiber = true
                    }
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
    s.has_svc
        && !(s.has_park_seam
            || s.has_fiber
            || s.has_thread
            || s.has_coro
            || s.has_instantiate
            || s.has_gc)
}

/// Lower every function, or `None` if any uses an op outside this slice's subset.
pub fn compile_module(funcs: &[Func]) -> Option<Compiled> {
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
    // `gc.roots` scans only the **calling vCPU's** continuation (its stack, fibers, coroutines), so a
    // module that also spawns threads could hold roots in a sibling vCPU we wouldn't scan — reject
    // that combination (fall back) to stay sound. `gc.roots` + fibers / coroutines is fine (those
    // continuations *are* scanned).
    //
    // §3.6 (I36 slice 1): a **serving** module is admitted natively only when no handler could
    // park or unwind mid-dispatch ([`serve_qualifies`]) — any park-capable seam anywhere in the
    // module (futex waits / threads, fibers, coroutines, nested instantiate, setjmp/longjmp — a
    // `longjmp` out of a handler would unwind past the serve linkage — blocking stream reads,
    // spawn-bound imports, gc.roots) falls the whole module back to the tree-walk oracle, whose
    // serve arm has the fiber-park machinery (slice 5b).
    if (s.has_coro && (s.has_fiber || s.has_thread))
        || (s.has_instantiate && s.has_fiber)
        || (s.has_gc && s.has_thread)
        || (s.has_svc
            && (s.has_park_seam
                || s.has_fiber
                || s.has_thread
                || s.has_coro
                || s.has_instantiate
                || s.has_gc))
    {
        return None;
    }

    let arities: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    let mut progs = Vec::with_capacity(funcs.len());
    for f in funcs {
        progs.push(compile_func(f, &arities)?);
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

fn compile_func(f: &Func, arities: &[usize]) -> Option<Program> {
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
    for (bi, b) in f.blocks.iter().enumerate() {
        block_pc[bi] = ops.len() as u32;
        let g = |local: u32| base[bi] + local; // operand: block-local index -> frame slot
        let mut local = b.params.len() as u32;
        for inst in &b.insts {
            let dst = base[bi] + local;
            local += inst.result_count(arities) as u32;
            ops.push(compile_inst(inst, dst, base[bi], &g)?);
        }
        // Terminator -> edge copies (block-local src in this block -> first slots of target) + jump.
        let edge = |bidx: usize, args: &[u32]| -> Edge {
            let copies = args
                .iter()
                .enumerate()
                .map(|(i, a)| (g(*a), base[bidx] + i as u32))
                .collect();
            (copies, bidx as u32) // block index; patched to entry pc below
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
                ops.push(Op::BrIf {
                    cond: g(*cond),
                    then_copies,
                    then_pc: tt,
                    else_copies,
                    else_pc: et,
                });
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
    }
    // Debug reverse map (Slice 1c-3): each block lays out `insts.len()` instruction ops at
    // `[block_pc[bi], +insts.len())` then exactly one terminator op. Instruction ops map to their
    // `(block, inst)`; the terminator op maps to `(block, insts.len() | SRC_TERM)` — flagged so
    // `cur_ir_pc` skips it (non-steppable) while `vm_trap_bt` can still name a terminator-trap site
    // (`unreachable`). The later target-patch only rewrites jump fields, not the op order, so this
    // index stays valid.
    let mut src: Vec<Option<(u32, u32)>> = vec![None; ops.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        let base_pc = block_pc[bi] as usize;
        for i in 0..b.insts.len() {
            src[base_pc + i] = Some((bi as u32, i as u32));
        }
        src[base_pc + b.insts.len()] = Some((bi as u32, b.insts.len() as u32 | SRC_TERM));
    }

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
        Inst::PtrAdd { a, b } => Op::PtrAdd {
            dst,
            a: g(*a),
            b: g(*b),
        },
        Inst::PtrCast { a, .. } => Op::PtrCast { dst, a: g(*a) },
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
                },
                // op 6/7 = spawn_coroutine_module / spawn_demand_coroutine_module: a coroutine child
                // running a granted `Module` (the first arg); the carve args (entry/off/size_log2/fuel)
                // follow. op 7 demand-pages the child's window (data segments supplied lazily).
                (cap_id::INSTANTIATOR, op @ (6 | 7)) if args.len() >= 4 => {
                    Op::SpawnCoroutineModule {
                        handle: g(*handle),
                        module: g(args[0]),
                        entry: g(args[1]),
                        off: g(args[2]),
                        size_log2: g(args[3]),
                        dst,
                        demand: op == 7,
                    }
                }
                // §3.6 (I36 slice 2) — child_offer (op 14): mint a live-callee offer over a running
                // child's export. The mint needs the child's live env, so the op surfaces to the
                // driver; the compile only marshals `(child, export)`.
                (cap_id::INSTANTIATOR, 14) if args.len() >= 2 => Op::ChildOffer {
                    handle: g(*handle),
                    child: g(args[0]),
                    export: g(args[1]),
                    dst,
                },
                // §14 cooperative coroutine round-trip — spawn_coroutine (op 2) / spawn_demand_coroutine
                // (op 4, window starts unmapped) / resume / yield.
                (cap_id::INSTANTIATOR, op @ (2 | 4)) if args.len() >= 3 => Op::SpawnCoroutine {
                    handle: g(*handle),
                    entry: g(args[0]),
                    off: g(args[1]),
                    size_log2: g(args[2]),
                    dst,
                    demand: op == 4,
                },
                (cap_id::INSTANTIATOR, 3) if args.len() >= 2 => Op::CoResume {
                    handle: g(*handle),
                    ch: g(args[0]),
                    value: g(args[1]),
                    dst,
                },
                (cap_id::YIELDER, 0) if !args.is_empty() => Op::CoYield {
                    handle: g(*handle),
                    value: g(args[0]),
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
                (cap_id::INSTANTIATOR, _) | (cap_id::YIELDER, _) => return None,
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
                // Generic synchronous powerbox dispatch (Stream/Clock/Memory/host-fn/JIT compile/…).
                _ => Op::CapCall {
                    type_id: *type_id,
                    op: *op,
                    handle: g(*handle),
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                    results: sig.results.clone().into(),
                },
            }
        }
        // §7 reflection — synchronous self-powerbox queries (no scheduler/fiber); reuse the host's
        // `self_dispatch`, the same path the tree-walker and the JIT thunk take.
        Inst::CapSelfCount => Op::CapSelfCount { dst },
        Inst::CapSelfAttest => Op::CapSelfAttest { dst },
        Inst::CapSelfGet { idx } => Op::CapSelfGet { idx: g(*idx), dst },
        Inst::CapSelfResolve { name_ptr, name_len } => Op::CapSelfResolve {
            name_ptr: g(*name_ptr),
            name_len: g(*name_len),
            dst,
        },
        Inst::CapSelfLabel {
            handle,
            buf_ptr,
            buf_cap,
        } => Op::CapSelfLabel {
            handle: g(*handle),
            buf_ptr: g(*buf_ptr),
            buf_cap: g(*buf_cap),
            dst,
        },
        // §12 fibers — cooperative continuation switching, driven by the bytecode driver (no M:N
        // pool, no DPOR; single-vCPU). `cont.new` registers a pending fiber, `cont.resume` switches
        // in (two results), `suspend` switches back (one result).
        Inst::ContNew { func, sp } => Op::ContNew {
            func: g(*func),
            sp: g(*sp),
            dst,
        },
        Inst::ContResume { k, arg } => Op::ContResume {
            k: g(*k),
            arg: g(*arg),
            dst,
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
            args: [g(*handle)].into(),
            dst,
            results: [ValType::I32].into(),
        },
        // §12.8 4A.5: serviced from the running `Vm`'s region base (the reference `eval_inst` has no
        // context), so it gets a dedicated op rather than the `Eval` fallback.
        Inst::DurableShadowBase => Op::DurableShadowBase { dst },
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, mc.size_log2, back);
        if seed_data {
            mm.init_data(&m.data);
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
}

impl SharedProgram {
    /// Compile `m` once (`None` if it uses an op outside the engine's subset).
    pub fn compile(m: &Module) -> Option<SharedProgram> {
        let c = compile_module(&m.funcs)?;
        let n_funcs = c.progs.len();
        Some(SharedProgram {
            source: std::sync::Arc::new(ModuleSource::new(c)),
            n_funcs,
            mem_size_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone(),
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
        if func as usize >= self.n_funcs {
            return Err(Trap::Malformed);
        }
        // A fresh natural dispatch table over the shared compiled source (cheap: an `Arc` clone + the
        // slot vector) — the cross-tier reactor carries no §22 install state between calls.
        let dom = Domain::child(self.source.clone(), SharedSlots::new(self.n_funcs, 0, 0));
        let mut mem = self.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back);
            if seed_data {
                mm.init_data(&self.data);
            }
            mm
        });
        run(dom, func, args, fuel, &mut mem, host)
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
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
        let c = compile_module(&m.funcs)?;
        let dom = Domain::new(c, table_log2);
        Some(VcpuProgram {
            dom,
            mem_size_log2: m.memory.as_ref().map(|mc| mc.size_log2),
            data: m.data.clone(),
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
        let c = compile_module(&m.funcs)?;
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
        })
    }

    /// Enable wasm-JIT tier-up: a direct `Call` to a function `f` with `eligible[f] == true` surfaces
    /// as [`VcpuEvent::TierUp`] for the `frame` caller to service. `None` (the default) interprets
    /// everything — the faithful [`Reactor`] substitute.
    pub fn with_jit_eligible(mut self, eligible: std::sync::Arc<[bool]>) -> VcpuReactor {
        self.eligible = Some(eligible);
        self
    }

    /// Run `func(args)` on the live window for one frame, servicing host caps inline against `host`.
    /// A [`VcpuEvent::TierUp`] is handed to `service(func, argv)` — return the callee's i64 result
    /// slots (or an `Err(Trap)` to propagate the emitted region's trap). With no eligibility set,
    /// `service` is never called. The window persists after the call (reclaimed for the next frame).
    pub fn frame<F>(
        &mut self,
        func: FuncIdx,
        args: &[Value],
        host: &std::sync::Mutex<Host>,
        mut service: F,
    ) -> Result<Vec<Value>, Trap>
    where
        F: FnMut(u32, &[i64]) -> Result<Vec<i64>, Trap>,
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
            result = loop {
                match vcpu.run() {
                    VcpuEvent::Done(v) => break Ok(v),
                    VcpuEvent::Trapped(t) => break Err(t),
                    VcpuEvent::TierUp { func, argv } => match service(func, &argv) {
                        Ok(vals) => vcpu.deliver_tierup(&vals),
                        Err(t) => vcpu.deliver_tierup_trap(t),
                    },
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
    TierUp { func: u32, argv: Box<[i64]> },
    /// `thread.spawn`: start `func(sp, arg)` as a new vCPU, then call [`Vcpu::deliver_handle`] with the
    /// handle the guest will `join` it by (the host assigns handles densely per spawner: 0, 1, …).
    Spawn { func: u32, sp: i64, arg: i64 },
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
    JitInvoke {
        handle: i32,
        code: i32,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
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
    /// A trap to surface on the next `run` (a joined child trap propagates to the joiner).
    trap: Option<Trap>,
    /// **wasm-JIT tier-up eligibility** (browser wasm-JIT threads slice). When set, `jit_eligible[f]`
    /// means function `f`'s whole reachable region is JIT-compilable and suspension-free, so a direct
    /// `Call` to it is surfaced as a [`VcpuEvent::TierUp`] — the host runs the emitted `f{f}` on the
    /// Worker (top-level caller, so a guest trap is a catchable `RuntimeError`) and delivers the
    /// result back via [`deliver_tierup`](Vcpu::deliver_tierup). `None` ⇒ everything interprets, as
    /// before this seam existed. The engine stays wasm-agnostic: it consults only this bitmap; the
    /// embedder computes it (e.g. from `svm_wasmjit::analyze`).
    jit_eligible: Option<std::sync::Arc<[bool]>>,
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
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// A `thread.spawn`ed **child** vCPU: shares `back` but does **not** re-seed (the window is already
    /// live with the root's image + every vCPU's writes).
    pub fn new_child(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog
            .mem_size_log2
            .map(|sl| Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back));
        Vcpu::with_mem(prog, func, args, mem, Host::new())
    }

    fn with_mem(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        mem: Option<Mem>,
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        if func as usize >= prog.dom.source.primary().progs.len() {
            return Err(Trap::Malformed);
        }
        Ok(Vcpu {
            vt: VTask::new(&prog.dom.source.primary(), func as usize, args)?,
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
            trap: None,
            jit_eligible: None,
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
            trap: None,
            jit_eligible: None,
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
            );
            match stop {
                // §3.6 (I36 slice 2): live calls / svc.wait / child_offer need the cooperative
                // scheduler's waker topology (`drive`); on this single-vCPU driver nothing could
                // ever wake or mint them — fail closed rather than hang. Unreachable through the
                // compile (op-14 implies the drive path); requires a hand-wired live cap.
                Ok(VcpuStop::LiveCall { .. })
                | Ok(VcpuStop::SvcWait)
                | Ok(VcpuStop::ChildOffer { .. }) => return VcpuEvent::Trapped(Trap::ThreadFault),
                Err(t) => return VcpuEvent::Trapped(t),
                Ok(VcpuStop::Done(vals)) => return VcpuEvent::Done(vals),
                Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                }) => {
                    self.pending_tierup = Some((dst, results));
                    return VcpuEvent::TierUp { func, argv };
                }
                Ok(VcpuStop::Spawn { func, sp, arg, dst }) => {
                    if func as usize >= dom.source.primary().progs.len() {
                        return VcpuEvent::Trapped(Trap::Malformed);
                    }
                    self.pending = Some(dst);
                    return VcpuEvent::Spawn { func, sp, arg };
                }
                Ok(VcpuStop::Join { handle, dst }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Join { handle };
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
                    return VcpuEvent::JitInvoke {
                        handle: h,
                        code,
                        argv,
                        params,
                        results,
                    };
                }
                // §14 `spawn_coroutine_module` — serviced **internally** against this vCPU's own
                // powerbox (which resolved the `Instantiator` authority in-Vm to reach here): resolve
                // the granted module, build the inline `Coro`, then loop (the coroutine runs inline via
                // `resume`, no host round-trip). A bad carve/entry/module sets `EINVAL`/traps in place.
                Ok(VcpuStop::SpawnCoroutineModule {
                    ibase,
                    isize: isz,
                    mh,
                    entry,
                    off,
                    size_log2,
                    dst,
                    demand,
                }) => {
                    self.service_coroutine_module(
                        ibase, isz, mh, entry, off, size_log2, demand, dst,
                    );
                    if let Some(t) = self.trap.take() {
                        return VcpuEvent::Trapped(t);
                    }
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
                }) => {
                    if let Some(ev) =
                        self.event_instantiate(ibase, isz, entry, off, size_log2, quota, dst)
                    {
                        return ev;
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
                }) => {
                    match self
                        .event_instantiate_module(ibase, isz, mh, entry, off, size_log2, quota, dst)
                    {
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

    /// Validate + prepare a §14 `instantiate` (op 0, same-module child) and produce its
    /// [`VcpuEvent::Instantiate`], or land `-EINVAL` in place (`None`) on a bad entry/carve —
    /// identical checks to the cooperative and parallel drivers' arms.
    #[allow(clippy::too_many_arguments)]
    fn event_instantiate(
        &mut self,
        ibase: u64,
        isize: u64,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
        dst: u32,
    ) -> Option<VcpuEvent> {
        let c0 = self.prog.dom.source.primary();
        let ok_entry = c0.sigs.get(entry as usize).is_some_and(|(p, r)| {
            r[..] == [ValType::I64]
                && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
        });
        let child_size = if (0..64).contains(&size_log2) {
            1u64 << size_log2
        } else {
            0
        };
        let off_u = off as u64;
        let fits = child_size != 0
            && child_size <= isize
            && off_u & (child_size - 1) == 0
            && off_u.checked_add(child_size).is_some_and(|e| e <= isize);
        if !ok_entry || !fits {
            self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            return None;
        }
        // Window-relative carve (`window.base()` is 0 for this path's top-level region windows; the
        // term keeps exact parity with the drive/parallel arms' backing-absolute math).
        let pbase = self.mem.as_ref().map_or(0, |m| m.window.base());
        let carve = pbase + ibase + off_u;
        let fuel = if quota <= 0 {
            self.fuel
        } else {
            (quota as u64).min(self.fuel)
        };
        self.pending = Some(dst);
        Some(VcpuEvent::Instantiate {
            module: 0,
            entry: entry as u32,
            carve,
            size_log2: size_log2 as u8,
            fuel,
        })
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
        dst: u32,
    ) -> Result<Option<VcpuEvent>, Trap> {
        // Resolve the granted module from the run's powerbox (the shared one when attached).
        let (cfuncs, cmem_log2, cdata) = match self.shared_host {
            Some(m) => {
                let g = m.lock().unwrap_or_else(|e| e.into_inner());
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
            .is_some_and(|(p, r)| {
                r[..] == [ValType::I64]
                    && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
            });
        let child_size = if (0..64).contains(&size_log2) {
            1u64 << size_log2
        } else {
            0
        };
        let off_u = off as u64;
        let fits = child_size != 0
            && child_size <= isize
            && off_u & (child_size - 1) == 0
            && off_u.checked_add(child_size).is_some_and(|e| e <= isize);
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
        let fuel = if quota <= 0 {
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

    /// Build a §14 `spawn_coroutine_module` coroutine **inline** in this vCPU's coroutine set (the
    /// resumable-path counterpart of the parallel driver's arm): resolve the granted module from this
    /// vCPU's own powerbox, compile + push it to the shared source, materialize its data segments into
    /// the carve (demand-page for op 7), and register the `Coro` (Yielder-only powerbox) — its handle
    /// written to `dst`. `EINVAL` on a bad entry/carve/memory mismatch; a resolve failure sets `trap`.
    #[allow(clippy::too_many_arguments)]
    fn service_coroutine_module(
        &mut self,
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        demand: bool,
        dst: u32,
    ) {
        let isz = isize;
        // Resolve the granted module from the run's powerbox — the shared one when attached, else
        // this vCPU's own (forged/closed handle → trap).
        let resolved = match self.shared_host {
            Some(m) => {
                let g = m.lock().unwrap_or_else(|e| e.into_inner());
                g.resolve_module(mh)
                    .map(|g| (g.funcs.clone(), g.memory_log2, g.data.clone()))
            }
            None => self
                .host
                .resolve_module(mh)
                .map(|g| (g.funcs.clone(), g.memory_log2, g.data.clone())),
        };
        let (cfuncs, cmem_log2, cdata) = match resolved {
            Ok(parts) => parts,
            Err(t) => {
                self.trap = Some(t);
                return;
            }
        };
        let child_compiled = match compile_module(&cfuncs) {
            Some(c) => c,
            None => {
                self.trap = Some(Trap::Malformed);
                return;
            }
        };
        let ok_entry = child_compiled
            .sigs
            .get(entry as usize)
            .is_some_and(|(p, r)| p[..] == [ValType::I64] && r[..] == [ValType::I64]);
        let child_size = if (0..64).contains(&size_log2) {
            1u64 << size_log2
        } else {
            0
        };
        let off_u = off as u64;
        let fits = child_size != 0
            && child_size <= isz
            && off_u & (child_size - 1) == 0
            && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
        let mod_ok = cmem_log2 == Some(size_log2 as u8);
        if !ok_entry || !fits || !mod_ok {
            self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            return;
        }
        let pbase = self.mem.as_ref().map_or(0, |m| m.window.base());
        let abs_base = pbase + ibase + off_u;
        let child_mem = {
            if let Some(m) = self.mem.as_ref() {
                for d in cdata.iter() {
                    if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
                        for (k, &b) in d.bytes.iter().enumerate() {
                            m.set_byte(abs_base + d.offset + k as u64, b);
                        }
                    }
                }
            }
            self.mem.as_ref().map(|m| {
                let cm = m.nested_view(abs_base, size_log2 as u8);
                if demand {
                    cm.demand_page();
                }
                cm
            })
        };
        let mut child_host = Host::new();
        let cy = child_host.grant_yielder();
        // `self.prog` is a `Copy` shared reference — copy it out so the `&prog.dom` borrow is
        // independent of the `&mut self.vt` push below.
        let prog = self.prog;
        let progs_len = child_compiled.progs.len();
        let cm = prog.dom.source.push(child_compiled);
        let child_table = build_table_for(progs_len, 0, cm as u32);
        let cunit = prog.dom.source.get(cm);
        let mut child_vm =
            match cunit.and_then(|u| Vm::new(&u, entry as usize, &[Value::I64(cy as i64)]).ok()) {
                Some(v) => v,
                None => {
                    self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    return;
                }
            };
        child_vm.module = cm;
        self.vt.coroutines.push(Some(Coro {
            vm: child_vm,
            mem: child_mem,
            host: child_host,
            table: child_table,
            awaiting: None,
            fault_yields: demand,
            faulted_page: None,
        }));
        let h = (self.vt.coroutines.len() - 1) as i32;
        self.vt.active.set(dst, Reg::from_i32(h));
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
    pub fn deliver_jit_install(&mut self, funcs: Result<std::sync::Arc<[Func]>, Trap>) {
        let Some(PendingJit::Install { dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_install with no pending install");
        };
        let funcs = match funcs {
            Ok(f) => f,
            Err(t) => {
                self.trap = Some(t);
                return;
            }
        };
        let res = match compile_module(&funcs) {
            // Install into THIS vCPU's domain (== the shared one for a root; a §14 confined child —
            // which can't hold a Jit cap anyway — would only ever fill its own table).
            Some(unit) => match self
                .own_dom
                .as_ref()
                .unwrap_or(&self.prog.dom)
                .install(unit)
            {
                Some(slot) => slot as i64,
                None => super::ENOSPC,
            },
            None => {
                self.trap = Some(Trap::Malformed); // unit op outside coverage
                return;
            }
        };
        self.vt.active.set(dst, Reg::from_i64(res));
    }

    /// Deliver the authority check for a `JitUninstall`: `Err` propagates as a trap; `Ok(())` clears the
    /// shared table `slot` (`0` on success, `EINVAL` for a real-func / out-of-range / already-empty slot).
    pub fn deliver_jit_uninstall(&mut self, authorized: Result<(), Trap>) {
        let Some(PendingJit::Uninstall { slot, dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_uninstall with no pending uninstall");
        };
        if let Err(t) = authorized {
            self.trap = Some(t);
            return;
        }
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let n_real = dom.source.primary().progs.len();
        let res = if dom.uninstall(slot as usize, n_real) {
            0
        } else {
            super::EINVAL
        };
        self.vt.active.set(dst, Reg::from_i64(res));
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
            dom,
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

    /// Deliver a **trap** from a host-run [`VcpuEvent::JitInvoke`] unit (the emitted region hit a
    /// guest `unreachable` / memory fault / div-by-zero / out-of-fuel, surfaced to the host as a
    /// catchable `RuntimeError`). The vCPU traps on its next `run`, exactly as an interpreted invoke
    /// trap would (`deliver_jit_invoke` sets `self.trap` on the unit's `Err`).
    pub fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
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
    let c = compile_module(&m.funcs)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
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
    let c = compile_module(&m.funcs)?;
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
    let super::MemAccess::Range { base, width, write } = super::access_of(ir_inst, vals, mem)
    else {
        return None;
    };
    let end = base.saturating_add(width as u64);
    watchpoints.iter().find_map(|(addr, len, kind)| {
        let w_end = addr.saturating_add(*len);
        (base < w_end && *addr < end && kind.fires_on(write)).then_some((base, write))
    })
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
        Ok(Outcome::ContResume { kh, arg, dst }) => {
            let k = kh as usize;
            let target = match fibers.get_mut(k) {
                Some(slot @ FiberState::Pending { .. }) => {
                    let (funcref, sp) = match std::mem::replace(slot, FiberState::Running) {
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
                    match std::mem::replace(slot, FiberState::Running) {
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
        // Threads / wait / notify / instantiate / coroutine / tier-up — a scheduler seam the caller
        // applies (single-vCPU `DebugRun` rejects them; the scheduled engine dispatches its subset).
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
}

impl FrameReader<'_> {
    /// Call-stack depth (running activation + suspended callers).
    fn depth(&self) -> usize {
        self.vm.stack.len() + 1
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

    /// Block-local SSA value `idx` in the frame `depth` levels from the top, typed.
    fn value_in_frame(&self, depth: usize, idx: usize) -> Option<Value> {
        let (module, func, block, _inst, base) = self.frame_at(depth)?;
        if module != 0 {
            return None;
        }
        let off = *self.fn_block_base.get(func)?.get(block)? as usize;
        let ty = *self.fn_block_types.get(func)?.get(block)?.get(idx)?;
        Some(self.vm.regs[base + off + idx].to_value(ty))
    }

    /// Read a source variable by name in the frame `depth` levels from the top, resolving its `VarLoc`
    /// over the §6 debug info (SSA slot / window / fixed). `None` if unresolvable here.
    fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        let di = self.debug?;
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        if module != 0 {
            return None;
        }
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
    /// top; `None` for a promoted SSA scalar (no address) or an unresolvable name.
    fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        let di = self.debug?;
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        if module != 0 {
            return None;
        }
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
    /// A [`FrameReader`] over this single-vCPU session's `Vm` + debug metadata.
    fn reader(&self) -> FrameReader<'_> {
        FrameReader {
            vm: &self.vt.active,
            source: &self.source,
            mem: &self.mem,
            debug: self.debug.as_ref(),
            fn_block_base: &self.fn_block_base,
            fn_block_types: &self.fn_block_types,
        }
    }

    /// Open a debug session on `m`'s `func(args)`. `None` if the module is outside the engine's subset.
    pub fn new(m: &Module, func: FuncIdx, args: &[Value]) -> Option<DebugRun> {
        m.funcs.get(func as usize)?;
        let arities: Vec<usize> = m.funcs.iter().map(|g| g.results.len()).collect();
        // Slot base + value types per (function, block), so any frame on the call stack is readable.
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
        let c = compile_module(&m.funcs)?;
        let dom = Domain::new(c, 0);
        let mem = build_mem(m);
        let host = Host::new();
        let vt = VTask::new(&dom.source.primary(), func as usize, args).ok()?;
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

    /// Arm the "paused on a breakpoint" state so the next [`run_to`](DebugRun::run_to) steps past the
    /// current op before scanning — used after a `seek`/replay lands exactly on a breakpoint, so a
    /// forward resume makes progress instead of re-reporting the same stop.
    pub fn arm_breakpoint_skip(&mut self) {
        self.at_bp = true;
    }

    /// Execute **exactly one op** (advancing the clock), for replay-based `seek`. Returns `false` once
    /// the run has finished (its result is then available via [`result`](DebugRun::result)). Unlike the
    /// stepping verbs it does not skip unmapped ops or honor breakpoints — it is the raw time quantum.
    pub fn tick(&mut self, fuel: &mut u64) -> bool {
        if self.done.is_some() {
            return false;
        }
        self.at_bp = false;
        let Self {
            source,
            table,
            mem,
            host,
            vt,
            fibers,
            done,
            op_clock,
            ..
        } = self;
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
            funcs,
            watchpoints,
            last_watch,
            ..
        } = self;
        // Step past the breakpoint we last reported, so a re-entry makes progress (loop bodies).
        if *at_bp {
            *at_bp = false;
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
                // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
                FiberStep::Other(_) => {
                    *done = Some(Err(Trap::Malformed));
                    return None;
                }
            }
        }
        loop {
            if let Some(pc) = vt.active.cur_ir_pc(source) {
                if bps.contains(&pc) {
                    *at_bp = true;
                    return Some(pc);
                }
                // Watchpoint: stop *before* an op that touches a watched window range (the access
                // hasn't applied — step once to observe the new bytes). Skipped when none are armed.
                if !watchpoints.is_empty() && pc.module == 0 {
                    if let Some(hit) = watch_hit_before(
                        &vt.active,
                        &*mem,
                        funcs,
                        fn_block_base,
                        watchpoints,
                        pc.func,
                        pc.block,
                        pc.inst,
                    ) {
                        *last_watch = Some(hit);
                        *at_bp = true;
                        return Some(pc);
                    }
                }
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
            ..
        } = self;
        *at_bp = false; // a step leaves the breakpoint-paused state
        loop {
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
                // A scheduler seam (threads/instantiate/…) is out of the single-vCPU debug scope.
                FiberStep::Other(_) => {
                    *done = Some(Err(Trap::Malformed));
                    return None;
                }
            }
            let depth = vt.active.stack.len() + 1;
            if max_depth.is_none_or(|m| depth <= m) {
                if let Some(pc) = vt.active.cur_ir_pc(source) {
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
        let d = self.depth();
        self.step_to(Some(d), fuel)
    }

    /// **Step out**: run until the current function returns, stopping at the op in the caller it
    /// returned to (from the outermost frame, runs to completion). The counterpart of
    /// `Inspector::step_out`.
    pub fn step_out(&mut self, fuel: &mut u64) -> Option<super::IrPc> {
        let d = self.depth();
        self.step_to(Some(d.saturating_sub(1)), fuel)
    }

    /// Number of live call frames at the current stop (callers + the running activation) — the depth a
    /// DAP `stackTrace` would report.
    pub fn depth(&self) -> usize {
        self.reader().depth()
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
    /// pointer. Errs if the range is unmapped or the module has no memory.
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match self.mem.as_ref() {
            Some(m) => m.read_window(addr, len),
            None => Err(Trap::Malformed),
        }
    }

    /// The running frame's block-local SSA value `idx` ([`value_in_frame`] at depth 0).
    pub fn value(&self, idx: usize) -> Option<Value> {
        self.value_in_frame(0, idx)
    }

    /// The run result once finished (`None` while still running).
    pub fn result(&self) -> Option<&Result<Vec<Value>, Trap>> {
        self.done.as_ref()
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
    /// A thread reached an op outside the debug scheduler's subset — `memory.wait`/`notify`, fibers,
    /// `instantiate`, coroutines, tier-up. This program can't be debugged multithreaded here.
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

/// One scheduled vCPU under the multi-vCPU debugger.
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
    /// This vCPU's `thread.spawn` children (handle = index → global task index; `None` = joined).
    threads: Vec<Option<usize>>,
    state: DbgTaskState,
    /// Paused on a just-reported breakpoint — step one op past it before the next scan makes progress
    /// (the per-task analogue of [`DebugRun::at_bp`], so a loop-body breakpoint re-fires each iteration).
    at_bp: bool,
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
/// so a fiber migrates across vCPUs — D57). Anything still outside the subset (`instantiate`,
/// coroutines) surfaces as [`SchedStop::Declined`].
pub struct ScheduledDebugRun {
    source: std::sync::Arc<ModuleSource>,
    table: SharedSlots,
    mem: Option<Mem>,
    host: Host,
    tasks: Vec<DbgTask>,
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
fn dbg_spawn(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    func: u32,
    sp: i64,
    arg: i64,
    dst: u32,
    source: &ModuleSource,
) -> Result<(), Trap> {
    let primary = source.primary();
    if func as usize >= primary.progs.len() {
        return Err(Trap::Malformed);
    }
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // thread bomb
    }
    let vt = VTask::new(&primary, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt,
        threads: Vec::new(),
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

/// Pick the next thread to run: the lowest-index runnable one. If none is runnable, advance the futex
/// `clock` to the earliest `memory.wait` deadline and wake every timed-out waiter (`WAIT_TIMED_OUT`),
/// then retry. `None` only on a true deadlock (no runnable thread and no waiter) — mirrors `drive`.
fn dbg_pick_runnable(tasks: &mut [DbgTask], clock: &mut u64) -> Option<usize> {
    loop {
        if let Some(i) = tasks
            .iter()
            .position(|t| matches!(t.state, DbgTaskState::Runnable))
        {
            return Some(i);
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
    /// bytecode engine's subset (`compile_module` declines it).
    pub fn new(m: &Module, func: FuncIdx, args: &[Value]) -> Option<ScheduledDebugRun> {
        m.funcs.get(func as usize)?;
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
        let c = compile_module(&m.funcs)?;
        let dom = Domain::new(c, 0);
        let mem = build_mem(m);
        let host = Host::new();
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
                state: DbgTaskState::Runnable,
                at_bp: false,
            }],
            fibers: Vec::new(),
            fn_block_base,
            fn_block_types,
            debug: m.debug_info.clone(),
            funcs: std::sync::Arc::from(m.funcs.clone()),
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
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
            fibers,
            funcs,
            breakpoints,
            watchpoints,
            last_watch,
            fn_block_base,
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
            // Prefer the stepping thread while it is runnable (so a step stays on it and a step-over
            // runs its own call), else the lowest-index runnable thread (advancing the futex clock to
            // wake a waiter when the set is stuck; unblocks a stepped `join`/`wait`).
            let ti = match step {
                Some((st, _)) if matches!(tasks[st].state, DbgTaskState::Runnable) => st,
                _ => match dbg_pick_runnable(tasks, clock) {
                    Some(i) => i,
                    None => return SchedStop::Blocked,
                },
            };
            // Pre-op stop checks (breakpoint / watchpoint), skipped for a thread that just reported (it
            // must make progress off its current op first, so a loop-body stop re-fires each iteration).
            if !tasks[ti].at_bp {
                if let Some(pc) = tasks[ti].vt.active.cur_ir_pc(source) {
                    if breakpoints.contains(&pc) {
                        tasks[ti].at_bp = true;
                        *stopped = Some(ti);
                        *focus = ti;
                        return SchedStop::Break {
                            pc,
                            reason: SchedBreak::Breakpoint,
                        };
                    }
                    if !watchpoints.is_empty() && pc.module == 0 {
                        if let Some((addr, write)) = watch_hit_before(
                            &tasks[ti].vt.active,
                            mem,
                            funcs,
                            fn_block_base,
                            watchpoints,
                            pc.func,
                            pc.block,
                            pc.inst,
                        ) {
                            *last_watch = Some((addr, write));
                            tasks[ti].at_bp = true;
                            *stopped = Some(ti);
                            *focus = ti;
                            return SchedStop::Break {
                                pc,
                                reason: SchedBreak::Watchpoint { addr, write },
                            };
                        }
                    }
                }
            }
            let step_res =
                debug_advance_fiber(&mut tasks[ti].vt, fibers, source, table, fuel, mem, host);
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
                    Outcome::ThreadSpawn { func, sp, arg, dst } => {
                        *turn += 1;
                        if let Err(t) = dbg_spawn(tasks, ti, func, sp, arg, dst, source) {
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
                        dbg_wait(tasks, ti, mem, *clock, base, expected, width, timeout, dst);
                    }
                    Outcome::MemoryNotify { base, count, dst } => {
                        *turn += 1;
                        dbg_notify(tasks, ti, base, count, dst);
                    }
                    // Instantiate / coroutine / tier-up — outside this slice.
                    _ => return SchedStop::Declined,
                },
            }
            // Post-op step target: the stepping thread reached a qualifying call depth at an instruction.
            if let Some((st, max_depth)) = step {
                if ti == st && matches!(tasks[st].state, DbgTaskState::Runnable) {
                    let depth = tasks[st].vt.active.stack.len() + 1;
                    if max_depth.is_none_or(|m| depth <= m) {
                        if let Some(pc) = tasks[st].vt.active.cur_ir_pc(source) {
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

    /// **Step over** the next source op: run any call it makes to completion (schedule advances only if
    /// the stepped thread blocks), landing at the next op at the same call depth.
    pub fn step_over(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self
            .stopped
            .map(|s| self.tasks[s].vt.active.stack.len() + 1);
        self.step_to(max, fuel)
    }

    /// **Step out** — run until the stepped thread's current function returns (one call depth shallower).
    pub fn step_out(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self
            .stopped
            .map(|s| (self.tasks[s].vt.active.stack.len() + 1).saturating_sub(1));
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
            fibers,
            turn,
            clock,
            ..
        } = self;
        let Some(ti) = dbg_pick_runnable(tasks, clock) else {
            return false; // no runnable thread and no waiter (deadlock) — can't advance
        };
        let step_res =
            debug_advance_fiber(&mut tasks[ti].vt, fibers, source, table, fuel, mem, host);
        tasks[ti].at_bp = false;
        *turn += 1;
        match step_res {
            FiberStep::Stepped => {}
            FiberStep::Finished(vals) => dbg_complete(tasks, ti, Ok(vals)),
            FiberStep::Trapped(t) => dbg_complete(tasks, ti, Err(t)),
            FiberStep::Other(outcome) => match outcome {
                Outcome::ThreadSpawn { func, sp, arg, dst } => {
                    if let Err(t) = dbg_spawn(tasks, ti, func, sp, arg, dst, source) {
                        dbg_complete(tasks, ti, Err(t));
                    }
                }
                Outcome::ThreadJoin { handle, dst } => dbg_join(tasks, ti, handle, dst),
                Outcome::MemoryWait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                } => dbg_wait(tasks, ti, mem, *clock, base, expected, width, timeout, dst),
                Outcome::MemoryNotify { base, count, dst } => {
                    dbg_notify(tasks, ti, base, count, dst)
                }
                _ => return false, // an unsupported op — stop the replay here
            },
        }
        !matches!(tasks[0].state, DbgTaskState::Done(_))
    }

    /// The current global turn (visible ops replayed so far) — the reverse-`seek` coordinate.
    pub fn op_turn(&self) -> u64 {
        self.turn
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

    /// A [`FrameReader`] over the **focused** thread's `Vm` (what `select_task` chose).
    fn reader(&self) -> FrameReader<'_> {
        FrameReader {
            vm: &self.tasks[self.focus].vt.active,
            source: &self.source,
            mem: &self.mem,
            debug: self.debug.as_ref(),
            fn_block_base: &self.fn_block_base,
            fn_block_types: &self.fn_block_types,
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

    /// Read `len` bytes from the shared guest window at `addr`.
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match self.mem.as_ref() {
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
    let c = compile_module(&m.funcs)?;
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
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
    },
    /// `cont.new`: register a fiber for `(funcref, sp)`, write its handle to `dst`, continue.
    ContNew {
        funcref: i32,
        sp: i64,
        dst: u32,
    },
    /// `cont.resume`: switch into fiber `kh` with `arg`; `(status, value)` land at `dst`/`dst+1`.
    ContResume {
        kh: i32,
        arg: i64,
        dst: u32,
    },
    /// `suspend`: hand `value` to the resumer; the parked fiber's `dst` receives the next resume arg.
    FiberSuspend {
        value: i64,
        dst: u32,
    },
    /// `thread.spawn`: spawn a vCPU running `func(sp, arg)`; its handle lands at `dst`.
    ThreadSpawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
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
    /// §14 `spawn_coroutine`: the Instantiator authority `(ibase, isize)` is resolved; build a child
    /// confined to `[ibase+off, +2^size_log2)` (`dst` ← handle or `EINVAL`).
    SpawnCoroutine {
        ibase: u64,
        isize: u64,
        entry: i64,
        off: i64,
        size_log2: i64,
        dst: u32,
        demand: bool,
    },
    /// §14 `spawn_coroutine_module`: like [`Outcome::SpawnCoroutine`], plus the resolved `Module`
    /// handle `mh` whose granted program the coroutine child runs (the driver resolves + compiles it).
    SpawnCoroutineModule {
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        dst: u32,
        demand: bool,
    },
    /// §14 `resume`: drive coroutine `ch` (authority already checked); `(status, value)` → `dst`.
    CoResume {
        ch: i32,
        value: i64,
        dst: u32,
    },
    /// §14 `yield`: this coroutine hands `value` to its resumer; the next resume's value → `dst`.
    CoYield {
        value: i64,
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

/// A §12 fiber's state in the driver's per-vCPU registry (handle = index). A durable run maintains the
/// per-context shadow-SP swap ([`shadow_switch`]) and, on freeze, flattens each `Parked` fiber into its
/// shadow region ([`freeze_drive`]); on thaw a flattened fiber is re-seeded as `Pending`.
enum FiberState {
    /// Created by `cont.new` but never resumed: starts by calling `funcref(sp, arg)`.
    Pending { funcref: i32, sp: i64 },
    /// Suspended mid-run; resuming delivers the new `arg` into `suspend_dst` and continues `vm`.
    Parked { vm: Vm, suspend_dst: u32 },
    /// Currently on the resume chain (active or an ancestor) — not independently resumable.
    Running,
    /// Returned; resuming again is a `FiberFault`.
    Done,
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
    /// §14 coroutine children this vCPU spawned (handle = index). `None` once finished. A coroutine
    /// is cooperative and driven *inline* by `resume` — never via the thread scheduler — so it lives
    /// here, not in the task set. (Coroutine modules are single-vCPU, no fibers/threads — see
    /// `compile_module` — so this and `chain` are never both non-empty.)
    coroutines: Vec<Option<Coro>>,
    /// DURABILITY.md §12.8 (D-fiber-cont option A): the root computation's (context 0's) saved durable
    /// shadow-stack pointer, swapped with the in-window active word ([`super::SHADOW_SP_OFF`]) on each
    /// fiber switch so a freeze poll spills into the *running* context's region. Only meaningful on a
    /// durable run; `super::SHADOW_BASE` (context 0's region base) otherwise.
    root_shadow_sp: u64,
}

impl VTask {
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<VTask, Trap> {
        Ok(VTask {
            active: Vm::new(c, entry, args)?,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            coroutines: Vec::new(),
            root_shadow_sp: super::SHADOW_BASE,
        })
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
            coroutines: Vec::new(),
            root_shadow_sp: root_sp,
        };
        match step_vcpu(&mut sub, fibers, fiber_sp, fiber_meta, dom, ctx, budget)? {
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

/// A §14 coroutine child: its own `Vm` continuation over a **confined** window (`nested_view`) and a
/// Yielder-only powerbox. Driven inline by `resume_coro` until it yields or returns. `awaiting` is the
/// `yield`'s result slot, set while suspended — the next `resume` writes the delivered value there.
/// `table` is the child's natural dispatch table: it maps into module 0 for a same-module coroutine
/// (op 2) or into the child's own pushed module index for a separate-module coroutine (op 6); the
/// `vm`'s `module` field selects which (no installed §22 units either way).
struct Coro {
    vm: Vm,
    mem: Option<Mem>,
    host: Host,
    table: SharedSlots,
    awaiting: Option<u32>,
    /// §14 **demand** coroutine (ops 4/7): its window starts fully unmapped, so an in-window access to
    /// an unsupplied page is a *recoverable* fault that suspends to the parent (which supplies the
    /// page) instead of trapping. A plain coroutine (ops 2/6) leaves this `false`.
    fault_yields: bool,
    /// Set while suspended on a recoverable page fault: the confined address to **supply** on the next
    /// `resume` (which then re-runs the rewound access). `None` otherwise.
    faulted_page: Option<u64>,
}

/// Why [`resume_coro`] returned: the coroutine yielded a value, hit a recoverable page fault (a
/// **demand** child — the parent must supply the page), or its function returned.
enum CoStop {
    Yield(i64),
    Fault(u64),
    Done(Vec<Value>),
}

/// Drive a coroutine child inline until it yields (`Yielder.yield` → [`Outcome::CoYield`]) or its
/// function returns. The child runs over its **own** confined `mem` and Yielder-only `host`; since it
/// holds no Instantiator, its own `spawn_coroutine`/`resume` resolve to `CapFault` inside
/// [`Vm::resume`] (never reaching here), and coroutine modules carry no fibers/threads — so the only
/// outcomes possible are `Done`/`Suspended`/`CoYield`. A child trap propagates to the resumer.
fn resume_coro(coro: &mut Coro, source: &ModuleSource, fuel: &mut u64) -> Result<CoStop, Trap> {
    // The coroutine child runs over its **own natural** table (built at spawn): it holds no `Jit`
    // cap, so it cannot reach installed §22 units (matching the tree-walker, where a coroutine child
    // gets a fresh `DomainTable::new(&cfuncs, 0)`). `coro.vm.module` selects its program (module 0 for
    // a same-module coroutine, its own pushed index for a separate-module one).
    //
    // A **demand** child (`fault_yields`) is stepped **one op at a time** (`budget = 1`): the budget
    // boundary persists the cursor *at* the next op before running it, so when that op faults the
    // cursor already points at it — re-running after the parent supplies the page retries exactly that
    // access, the §14 rewind, with **no** change to the hot loop (a plain coroutine runs unmetered).
    let budget = if coro.fault_yields { 1 } else { u64::MAX };
    loop {
        match coro.vm.resume(
            source,
            &coro.table,
            fuel,
            &mut coro.mem,
            &mut HostCell::Excl(&mut coro.host),
            budget,
        ) {
            Ok(Outcome::Done(vals)) => return Ok(CoStop::Done(vals)),
            Ok(Outcome::Suspended) => {} // budget exhausted (demand stepping) or normal — keep going
            Ok(Outcome::CoYield { value, dst }) => {
                coro.awaiting = Some(dst);
                return Ok(CoStop::Yield(value));
            }
            // A coroutine child is its own confined domain (no fibers/threads, holds no Instantiator),
            // so its `gc.roots` scans just its own continuation. Handle it inline and keep stepping.
            Ok(Outcome::GcRoots {
                lo,
                hi,
                mask,
                buf,
                cap,
                dst,
            }) => {
                let mut roots = std::collections::BTreeSet::new();
                {
                    let mut consider = |w: u64| {
                        let m = w & mask;
                        if m >= lo && m < hi {
                            roots.insert(m);
                        }
                    };
                    scan_vm_roots(&coro.vm, source, &mut consider);
                }
                let total = gc_write(&mut coro.mem, buf, cap, roots)?;
                coro.vm.set(dst, Reg::from_i64(total));
            }
            Ok(_) => return Err(Trap::FiberFault),
            // A demand child's *recoverable* in-window page fault suspends to the parent; an
            // out-of-window fault (`take_fault` → `None`) is a real trap that propagates.
            Err(Trap::MemoryFault) if coro.fault_yields => {
                match coro.mem.as_ref().and_then(|m| m.take_fault()) {
                    Some(addr) => return Ok(CoStop::Fault(addr)),
                    None => return Err(Trap::MemoryFault),
                }
            }
            Err(t) => return Err(t),
        }
    }
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
/// reaches installed units), to completion. An invoked unit is concurrency-/seam-free — the
/// tree-walker `CapFault`s if it parks, spawns, yields, or re-installs — so anything but a plain
/// return is an inert `CapFault`; a trap propagates to the invoker.
fn run_invoke(
    dom: &Domain,
    module: usize,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
) -> Result<Vec<Value>, Trap> {
    let unit = dom.source.get(module).ok_or(Trap::Malformed)?;
    let mut vm = Vm::new(&unit, 0, args)?;
    vm.module = module;
    loop {
        match vm.resume(&dom.source, &dom.table, fuel, mem, host, u64::MAX)? {
            Outcome::Done(vals) => return Ok(vals),
            Outcome::Suspended => {}
            _ => return Err(Trap::CapFault),
        }
    }
}

/// Why [`step_vcpu`] returned control to the scheduler: the vCPU finished, or it hit a multi-vCPU
/// (`thread.*` / `memory.*`) event the scheduler must service. Intra-vCPU fiber switches never reach
/// here — `step_vcpu` handles them against the vCPU's own registry.
enum VcpuStop {
    /// §3.6 (I36 slice 2): park this task on a live-call `ticket` against `callee` (see
    /// [`Outcome::LiveCall`] — the enqueue already happened in the op exec).
    LiveCall {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
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
    Done(Vec<Value>),
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): run the emitted `f{func}` region on the
    /// host, delivering its `n_results` results to absolute slot `dst` via `deliver_tierup`.
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
    },
    Spawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
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
    },
    /// §14 `Instantiator.spawn_coroutine_module` — the driver resolves + compiles the host-granted
    /// `Module` (`mh`), builds a coroutine `Coro` over it, and registers it in the spawner's coroutine
    /// set (thereafter `resume`d inline). Unlike `instantiate_module`, it creates no scheduler task.
    SpawnCoroutineModule {
        ibase: u64,
        isize: u64,
        mh: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        dst: u32,
        demand: bool,
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
            HostCell::Shared(m) => f(&mut m.lock().unwrap_or_else(|e| e.into_inner())),
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
fn step_vcpu(
    vt: &mut VTask,
    fibers: &mut Vec<FiberState>,
    fiber_sp: &mut Vec<u64>,
    fiber_meta: &mut Vec<(i32, i64)>,
    dom: &Domain,
    ctx: &mut RunCtx,
    budget: u64,
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
            Outcome::ContResume { kh, arg, dst } => {
                let k = kh as usize;
                // Claim fiber `k` from the **run-shared** registry: a pending fiber starts (call
                // `funcref(sp, arg)`), a parked one continues (the new `arg` becomes its `suspend`'s
                // result) — possibly one suspended on *another* vCPU (D57 migration). Anything else
                // (forged / already running on a vCPU / done) is inert.
                let target = match fibers.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let (funcref, sp) = match std::mem::replace(slot, FiberState::Running) {
                            FiberState::Pending { funcref, sp } => (funcref, sp),
                            _ => unreachable!(),
                        };
                        // Resolve the fiber entry through module 0's natural table + `fiber_sig`,
                        // exactly as `table_lookup` does — a forged/mistyped funcref is a
                        // `FiberFault`. Fibers are module-0 only (a unit cannot use `cont.*`).
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
                        match std::mem::replace(slot, FiberState::Running) {
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
            } => {
                return Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                })
            }
            Outcome::ThreadSpawn { func, sp, arg, dst } => {
                return Ok(VcpuStop::Spawn { func, sp, arg, dst })
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
            Outcome::Instantiate {
                ibase,
                isize: isz,
                entry,
                off,
                size_log2,
                quota,
                dst,
            } => {
                return Ok(VcpuStop::Instantiate {
                    ibase,
                    isize: isz,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
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
                })
            }
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
            // §14 coroutines are cooperative and driven **inline** here (never via the thread
            // scheduler), exactly as `run_inner` recurses for `resume`.
            Outcome::SpawnCoroutine {
                ibase,
                isize: isz,
                entry,
                off,
                size_log2,
                dst,
                demand,
            } => {
                let h = spawn_coroutine(
                    &mut vt.coroutines,
                    ctx.mem,
                    &dom.source.primary(),
                    entry,
                    (ibase, isz, off, size_log2),
                    demand,
                );
                vt.active.set(dst, Reg::from_i32(h));
            }
            // A separate-**module** coroutine spawn must compile + push the granted module (which
            // needs the mutable `Domain`), so it escapes to the driver; once built, it is `resume`d
            // inline like any coroutine.
            Outcome::SpawnCoroutineModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                dst,
                demand,
            } => {
                return Ok(VcpuStop::SpawnCoroutineModule {
                    ibase,
                    isize: isz,
                    mh,
                    entry,
                    off,
                    size_log2,
                    dst,
                    demand,
                })
            }
            Outcome::CoResume { ch, value, dst } => {
                // Take the coroutine; a forged/finished slot is an inert CapFault (propagates).
                let mut coro = vt
                    .coroutines
                    .get_mut(ch as usize)
                    .and_then(|c| c.take())
                    .ok_or(Trap::CapFault)?;
                if let Some(addr) = coro.faulted_page.take() {
                    // Resuming after a recoverable page fault: **supply** the page (keeping the
                    // parent's bytes), then re-run the rewound access — the value arg is unused.
                    if let Some(m) = coro.mem.as_ref() {
                        m.supply_page(addr);
                    }
                } else if let Some(yd) = coro.awaiting.take() {
                    coro.vm.set(yd, Reg::from_i64(value)); // deliver the resume value to the `yield`
                }
                match resume_coro(&mut coro, &dom.source, &mut *ctx.fuel)? {
                    CoStop::Yield(yv) => {
                        vt.coroutines[ch as usize] = Some(coro); // suspended — re-parked for next resume
                        vt.active.set(dst, Reg::from_i32(super::FIBER_SUSPENDED));
                        vt.active.set(dst + 1, Reg::from_i64(yv));
                    }
                    CoStop::Fault(addr) => {
                        // A demand child faulted: remember the page to supply, report (FAULTED, addr).
                        coro.faulted_page = Some(addr);
                        vt.coroutines[ch as usize] = Some(coro);
                        vt.active.set(dst, Reg::from_i32(super::CORO_FAULTED));
                        vt.active.set(dst + 1, Reg::from_i64(addr as i64));
                    }
                    CoStop::Done(vals) => {
                        // Finished — the slot stays `None` (a later resume is inert/CapFault).
                        vt.active.set(dst, Reg::from_i32(super::FIBER_RETURNED));
                        let v = vals.first().copied().unwrap_or(Value::I64(0));
                        vt.active.set(dst + 1, Reg::from_value(v));
                    }
                }
            }
            // A `Yielder.yield` only resolves (and thus only reaches a driver) inside an inline
            // coroutine child — `resume_coro` consumes it. At the top level the yielder handle is
            // ungranted, so `resume` CapFaults before producing this; treat any leak as a fault.
            Outcome::CoYield { .. } => return Err(Trap::FiberFault),
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
                        if let FiberState::Parked { vm, .. } = fib {
                            scan_vm_roots(vm, &dom.source, &mut consider);
                        }
                    }
                    for coro in vt.coroutines.iter().flatten() {
                        scan_vm_roots(&coro.vm, &dom.source, &mut consider);
                    }
                }
                let total = gc_write(ctx.mem, buf, cap, roots)?;
                vt.active.set(dst, Reg::from_i64(total));
            }
        }
    }
}

/// Build a §14 coroutine child confined to `[ibase+off, ibase+off+2^size_log2)` of the parent's
/// window, with a Yielder-only powerbox, and register it. Returns its handle, or `EINVAL` if the
/// entry signature / size / alignment is invalid (mirrors the tree-walker's validation).
fn spawn_coroutine(
    coroutines: &mut Vec<Option<Coro>>,
    mem: &Option<Mem>,
    c: &Compiled,
    entry: i64,
    // The Instantiator-relative carve geometry: `(holder base, holder size, offset, size_log2)`.
    carve: (u64, u64, i64, i64),
    // §14 op 4 `spawn_demand_coroutine`: start every page unmapped (lazy paging / fault-driven yield).
    demand: bool,
) -> i32 {
    let (ibase, isz, off, size_log2) = carve;
    // Coroutine entry is a fixed `(i64 yielder) -> (i64)`.
    let ok_entry = c
        .sigs
        .get(entry as u64 as usize)
        .is_some_and(|(p, r)| p[..] == [ValType::I64] && r[..] == [ValType::I64]);
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let fits = child_size != 0
        && child_size <= isz
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= isz);
    if !ok_entry || !fits {
        return super::EINVAL as i32;
    }
    // Holder-relative `ibase`/`off` → backing-absolute base (adds the holder's own window base, so
    // nesting composes); the child sees a zero-based `[0, child_size)` confined view.
    let abs_base = mem.as_ref().map_or(0, |m| m.window.base()) + ibase + off;
    let child_mem = mem.as_ref().map(|m| {
        let cm = m.nested_view(abs_base, size_log2 as u8);
        if demand {
            cm.demand_page(); // every page starts unmapped — faults suspend to us (lazy paging)
        }
        cm
    });
    let mut child_host = Host::new();
    let cy = child_host.grant_yielder(); // the child's handle to suspend back to us
    let child_vm = match Vm::new(c, entry as u64 as usize, &[Value::I64(cy as i64)]) {
        Ok(v) => v,
        Err(_) => return super::EINVAL as i32,
    };
    coroutines.push(Some(Coro {
        vm: child_vm,
        mem: child_mem,
        host: child_host,
        table: build_table(c.progs.len(), 0), // same-module coroutine: natural table over module 0
        awaiting: None,
        fault_yields: demand,
        faulted_page: None,
    }));
    (coroutines.len() - 1) as i32
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
    /// Parked on `memory.wait` at futex key `key` until notified or `deadline` (logical clock).
    BlockedWait {
        key: u64,
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
    let mut tasks: Vec<TaskSlot> = vec![TaskSlot {
        vt: VTask::new(&dom.source.primary(), entry as usize, args)?,
        threads: Vec::new(),
        env: None,
        state: TaskState::Runnable,
    }];
    // §14 `instantiate` children's confined environments (handle = `env` index). The root and its
    // `thread.spawn` siblings use the shared `mem`/`host`/`dom.table` instead (`env == None`).
    let mut extra_envs: Vec<ChildEnv> = Vec::new();
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
    let mut clock: u64 = 0;

    loop {
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
                host.frozen_fibers = freeze_drive(
                    &mut fibers,
                    &mut fiber_sp,
                    &mut fiber_meta,
                    &dom,
                    &mut ctx,
                    budget,
                )?;
            }
            return res;
        }
        // §3.6 (I36 slice 2) — settle wakes: a task parked on a live-call ticket wakes when the
        // callee's serve loop completed its dispatch; claiming the completion cell delivers the
        // reply (the tree-walker's cap_reply preference — a parked caller beats the cell).
        for t in &mut tasks {
            let hit = match &t.state {
                TaskState::BlockedTicket {
                    ticket,
                    callee,
                    dst,
                } => callee
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
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
        let Some(ti) = tasks
            .iter()
            .position(|t| matches!(t.state, TaskState::Runnable))
        else {
            // No runnable task: fire the earliest `wait` timeout, else it is a deadlock.
            let next = tasks
                .iter()
                .filter_map(|t| match t.state {
                    TaskState::BlockedWait { deadline, .. } => Some(deadline),
                    _ => None,
                })
                .min();
            match next {
                Some(d) => {
                    clock = clock.max(d);
                    for t in &mut tasks {
                        if let TaskState::BlockedWait { deadline, dst, .. } = t.state {
                            if deadline <= clock {
                                t.vt.active.set(dst, Reg::from_i32(super::WAIT_TIMED_OUT));
                                t.state = TaskState::Runnable;
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
            &mut fibers,
            &mut fiber_sp,
            &mut fiber_meta,
            &dom,
            &mut ctx,
            budget,
        );
        match stop {
            Err(trap) => complete(&mut tasks, ti, Err(trap)),
            Ok(VcpuStop::Done(vals)) => complete(&mut tasks, ti, Ok(vals)),
            // wasm-JIT tier-up is only enabled on the browser `Vcpu::run` path (`with_jit_eligible`);
            // the native drivers never set the eligibility bitmap, so it cannot occur here.
            Ok(VcpuStop::TierUp { .. }) => unreachable!("tier-up not enabled on the native driver"),
            // Blocking stdin is only ever set on an owned-host `Vcpu` (the interactive session), never
            // a scheduler task — same rationale as tier-up above.
            Ok(VcpuStop::StdinPark) => {
                unreachable!("blocking stdin not enabled on the scheduler driver")
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
                if let Some(k) = k {
                    for t in &mut tasks {
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
                    let sigs = callee
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .offer_shape(export)?;
                    match tasks[ti].env {
                        None => host.wire_live_impl(&callee, export, &sigs).ok(),
                        Some(pk) => extra_envs[pk]
                            .host
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .wire_live_impl(&callee, export, &sigs)
                            .ok(),
                    }
                });
                tasks[ti]
                    .vt
                    .active
                    .set(dst, Reg::from_i32(cap.unwrap_or(super::EINVAL as i32)));
            }
            Ok(VcpuStop::Spawn { func, sp, arg, dst }) => {
                if func as usize >= dom.source.primary().progs.len() {
                    complete(&mut tasks, ti, Err(Trap::Malformed));
                    continue;
                }
                let live = tasks
                    .iter()
                    .filter(|t| !matches!(t.state, TaskState::Done(_)))
                    .count();
                if live >= super::MAX_VCPUS {
                    complete(&mut tasks, ti, Err(Trap::ThreadFault)); // thread bomb
                    continue;
                }
                let child = VTask::new(
                    &dom.source.primary(),
                    func as usize,
                    &[Value::I64(sp), Value::I64(arg)],
                )?;
                let cidx = tasks.len();
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
            }) => {
                // Validate the child entry signature against module 0 (a same-module child): it
                // returns one `i64` and takes either its `Instantiator` (one `i64`) or its
                // `Instantiator`+`AddressSpace` (two) — its starter caps over its own window.
                let c0 = dom.source.primary();
                let want_as = c0
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                let ok_entry = c0.sigs.get(entry as usize).is_some_and(|(p, r)| {
                    r[..] == [ValType::I64]
                        && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
                });
                // The carve must be a power-of-two-aligned sub-window within `[0, isize)` — a child
                // gets only what the holder sub-allocates (§14/D19).
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
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
                    complete(&mut tasks, ti, Err(Trap::ThreadFault)); // instantiate bomb
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
                // over its *own* `[0, child_size)` window. These are its entry arguments.
                let mut child_host = Host::new();
                let cinst = child_host.grant_instantiator(0, child_size);
                let cas = child_host.grant_address_space(0, child_size);
                // §3.6: a same-module child serves over the shared program — its serve machinery
                // (enqueue admission, handler resolution) and any `child_offer` shape read the
                // domain's registered module, exactly the tree-walker's `self_module` handoff.
                child_host.self_module = match tasks[ti].env {
                    None => host.self_module.clone(),
                    Some(k) => extra_envs[k]
                        .host
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .self_module
                        .clone(),
                };
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
                        complete(&mut tasks, ti, Err(t));
                        continue;
                    }
                };
                // Compile the granted module to bytecode. A module using an op the engine can't lower
                // is the one place a guest-provided program outruns coverage (no tree-walker fallback
                // mid-run) — a `Malformed` trap, exactly as for `Jit.install`.
                let child_compiled = match compile_module(&cfuncs) {
                    Some(c) => c,
                    None => {
                        complete(&mut tasks, ti, Err(Trap::Malformed));
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
                    .is_some_and(|(p, r)| {
                        r[..] == [ValType::I64]
                            && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
                    });
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
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
                    complete(&mut tasks, ti, Err(Trap::ThreadFault));
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
                let mut child_host = Host::new();
                let cinst = child_host.grant_instantiator(0, child_size);
                let cas = child_host.grant_address_space(0, child_size);
                // §3.6: a separate-module child serves its OWN offers — enqueue admission,
                // handler resolution, and `child_offer` shape all read its module (tree-walk
                // lockstep: the spawn sets `self_module` from the grant).
                child_host.set_self_module(&cmodule);
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
            Ok(VcpuStop::SpawnCoroutineModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                dst,
                demand,
            }) => {
                // Resolve + compile the granted module (forged handle → CapFault; uncoverable op →
                // Malformed), exactly as for `instantiate_module`.
                let (cfuncs, cmem_log2, cdata) = match host.resolve_module(mh) {
                    Ok(g) => (g.funcs.clone(), g.memory_log2, g.data.clone()),
                    Err(t) => {
                        complete(&mut tasks, ti, Err(t));
                        continue;
                    }
                };
                let child_compiled = match compile_module(&cfuncs) {
                    Some(c) => c,
                    None => {
                        complete(&mut tasks, ti, Err(Trap::Malformed));
                        continue;
                    }
                };
                // A coroutine entry is `(i64 yielder) -> (i64)`; the carve must equal the module's
                // declared memory (§14 transparency).
                let ok_entry = child_compiled
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, r)| p[..] == [ValType::I64] && r[..] == [ValType::I64]);
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
                let mod_ok = cmem_log2 == Some(size_log2 as u8);
                if !ok_entry || !fits || !mod_ok {
                    tasks[ti]
                        .vt
                        .active
                        .set(dst, Reg::from_i32(super::EINVAL as i32));
                    continue;
                }
                let pbase = match tasks[ti].env {
                    None => mem.as_ref().map_or(0, |m| m.window.base()),
                    Some(k) => extra_envs[k].mem.as_ref().map_or(0, |m| m.window.base()),
                };
                let abs_base = pbase + ibase + off_u;
                // Build the child window and materialize the module's data segments into the carve
                // (as for `instantiate_module`).
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
                    pm.map(|m| {
                        let cm = m.nested_view(abs_base, size_log2 as u8);
                        if demand {
                            // op 7: every page starts unmapped — the materialized data segments are in
                            // the shared backing but **supplied lazily** as the child first touches them.
                            cm.demand_page();
                        }
                        cm
                    })
                };
                // A coroutine gets a Yielder-only powerbox (its single entry arg); it holds no
                // Instantiator, so its own spawn/resume CapFault inside `Vm::resume`.
                let mut child_host = Host::new();
                let cy = child_host.grant_yielder();
                let progs_len = child_compiled.progs.len();
                let cm = dom.source.push(child_compiled);
                let child_table = build_table_for(progs_len, 0, cm as u32);
                let cunit = dom.source.get(cm);
                let mut child_vm = match cunit
                    .and_then(|u| Vm::new(&u, entry as usize, &[Value::I64(cy as i64)]).ok())
                {
                    Some(v) => v,
                    None => {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                };
                child_vm.module = cm;
                // The coroutine lives in the spawning vCPU's coroutine set, driven inline by `resume`.
                tasks[ti].vt.coroutines.push(Some(Coro {
                    vm: child_vm,
                    mem: child_mem,
                    host: child_host,
                    table: child_table,
                    awaiting: None,
                    fault_yields: demand,
                    faulted_page: None,
                }));
                let h = (tasks[ti].vt.coroutines.len() - 1) as i32;
                tasks[ti].vt.active.set(dst, Reg::from_i32(h));
            }
            Ok(VcpuStop::Join { handle, dst }) => {
                let slot = match super::resolve_thread(&tasks[ti].threads, handle) {
                    Ok(s) => s,
                    Err(t) => {
                        complete(&mut tasks, ti, Err(t));
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
                            Err(t) => complete(&mut tasks, ti, Err(t)),
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
                // Re-read the value (the cooperative analogue of the futex compare-under-lock): if it
                // already changed, return not-equal; else park until notified or timed out.
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
                    tasks[ti].state = TaskState::BlockedWait {
                        key: base,
                        deadline: clock.saturating_add(timeout),
                        dst,
                    };
                }
            }
            Ok(VcpuStop::Notify { base, count, dst }) => {
                // Wake up to `count` waiters on `base`, lowest task index first (deterministic).
                let want = count as u32;
                let mut woken = 0u32;
                for t in &mut tasks {
                    if woken >= want {
                        break;
                    }
                    if let TaskState::BlockedWait { key, dst: wdst, .. } = t.state {
                        if key == base {
                            t.vt.active.set(wdst, Reg::from_i32(super::WAIT_WOKEN));
                            t.state = TaskState::Runnable;
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
                        complete(&mut tasks, ti, Err(t));
                        continue;
                    }
                };
                let res = match compile_module(&funcs) {
                    Some(unit) => match dom.install(unit) {
                        Some(slot) => slot as i64,
                        None => super::ENOSPC,
                    },
                    None => {
                        complete(&mut tasks, ti, Err(Trap::Malformed)); // unit op outside coverage
                        continue;
                    }
                };
                tasks[ti].vt.active.set(dst, Reg::from_i64(res));
            }
            Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                if let Err(t) = host.resolve_jit_domain(h) {
                    complete(&mut tasks, ti, Err(t)); // authority check
                    continue;
                }
                let n_real = dom.source.primary().progs.len();
                let res = if dom.uninstall(slot as usize, n_real) {
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
                        complete(&mut tasks, ti, Err(t));
                        continue;
                    }
                };
                let unit = match compile_module(&funcs) {
                    Some(u) => u,
                    None => {
                        complete(&mut tasks, ti, Err(Trap::Malformed));
                        continue;
                    }
                };
                // Arity-check the unit entry (func 0) against the call's (code-stripped) signature.
                let arity_ok = unit
                    .sigs
                    .first()
                    .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
                if !arity_ok {
                    complete(&mut tasks, ti, Err(Trap::CapFault));
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
                    &dom,
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
                        complete(&mut tasks, ti, Err(t));
                        continue;
                    }
                }
            }
        }
    }
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
        );
        match stop {
            // §3.6 (I36 slice 2): the serve/call/offer trio runs only on the cooperative
            // driver (`drive`); a serving module never reaches the parallel driver (the
            // qualification veto refuses svc + threads together) — fail closed if it somehow
            // does, rather than park unwakeably.
            Ok(VcpuStop::LiveCall { .. })
            | Ok(VcpuStop::SvcWait)
            | Ok(VcpuStop::ChildOffer { .. }) => return (Err(Trap::ThreadFault), mem),
            Err(trap) => return (Err(trap), mem),
            Ok(VcpuStop::Done(vals)) => return (Ok(vals), mem),
            // Tier-up is only enabled on the browser `Vcpu::run` path (`with_jit_eligible`).
            Ok(VcpuStop::TierUp { .. }) => unreachable!("tier-up not enabled on the native driver"),
            // Blocking stdin is only ever set on an owned-host `Vcpu` (the interactive session).
            Ok(VcpuStop::StdinPark) => {
                unreachable!("blocking stdin not enabled on the native driver")
            }
            Ok(VcpuStop::Spawn { func, sp, arg, dst }) => {
                if func as usize >= dom.source.primary().progs.len() {
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
                let child_vt = match VTask::new(
                    &dom.source.primary(),
                    func as usize,
                    &[Value::I64(sp), Value::I64(arg)],
                ) {
                    Ok(v) => v,
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
                    let g = host.lock().unwrap_or_else(|e| e.into_inner());
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
                    let g = host.lock().unwrap_or_else(|e| e.into_inner());
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
                    let g = host.lock().unwrap_or_else(|e| e.into_inner());
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
                    dom,
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
            }) => {
                // Validate the child entry signature against module 0 and the power-of-two-aligned
                // carve within `[0, isize)` — identical to the cooperative `drive` arm.
                let c0 = dom.source.primary();
                let want_as = c0
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, _)| p[..] == [ValType::I64, ValType::I64]);
                let ok_entry = c0.sigs.get(entry as usize).is_some_and(|(p, r)| {
                    r[..] == [ValType::I64]
                        && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
                });
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
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
            }) => {
                // Resolve + clone the granted module under the host lock (a forged/closed/wrong-type
                // handle is an inert CapFault → trap).
                let (cfuncs, cmem_log2, cdata) = {
                    let g = host.lock().unwrap_or_else(|e| e.into_inner());
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
                    .is_some_and(|(p, r)| {
                        r[..] == [ValType::I64]
                            && (p[..] == [ValType::I64] || p[..] == [ValType::I64, ValType::I64])
                    });
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
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
            // §14 `Instantiator.spawn_coroutine_module` (op 6 / demand op 7) — a separate-module
            // **coroutine**: resolve + compile + push the granted module (as for `instantiate_module`),
            // materialize its data segments, then build a `Coro` (Yielder-only powerbox) and register
            // it in **this** vCPU's coroutine set. Unlike instantiate, a coroutine is driven **inline**
            // by `resume` (no thread) — so once built it runs on this vCPU exactly as a same-module
            // coroutine already does in this driver; only the build (which needs the granted module)
            // escaped here.
            Ok(VcpuStop::SpawnCoroutineModule {
                ibase,
                isize: isz,
                mh,
                entry,
                off,
                size_log2,
                dst,
                demand,
            }) => {
                let (cfuncs, cmem_log2, cdata) = {
                    let g = host.lock().unwrap_or_else(|e| e.into_inner());
                    match g.resolve_module(mh) {
                        Ok(grant) => (grant.funcs.clone(), grant.memory_log2, grant.data.clone()),
                        Err(t) => return (Err(t), mem),
                    }
                };
                let child_compiled = match compile_module(&cfuncs) {
                    Some(c) => c,
                    None => return (Err(Trap::Malformed), mem),
                };
                // A coroutine entry is `(i64 yielder) -> (i64)`; the carve must equal the module's
                // declared memory (§14 transparency).
                let ok_entry = child_compiled
                    .sigs
                    .get(entry as usize)
                    .is_some_and(|(p, r)| p[..] == [ValType::I64] && r[..] == [ValType::I64]);
                let child_size = if (0..64).contains(&size_log2) {
                    1u64 << size_log2
                } else {
                    0
                };
                let off_u = off as u64;
                let fits = child_size != 0
                    && child_size <= isz
                    && off_u & (child_size - 1) == 0
                    && off_u.checked_add(child_size).is_some_and(|e| e <= isz);
                let mod_ok = cmem_log2 == Some(size_log2 as u8);
                if !ok_entry || !fits || !mod_ok {
                    vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                    continue;
                }
                let pbase = mem.as_ref().map_or(0, |m| m.window.base());
                let abs_base = pbase + ibase + off_u;
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
                    mem.as_ref().map(|m| {
                        let cm = m.nested_view(abs_base, size_log2 as u8);
                        if demand {
                            // op 7: every page starts unmapped — the data segments are in the shared
                            // backing but supplied lazily as the child first touches them.
                            cm.demand_page();
                        }
                        cm
                    })
                };
                // Yielder-only powerbox (its single entry arg): it holds no Instantiator, so its own
                // spawn/resume CapFaults inside `Vm::resume`.
                let mut child_host = Host::new();
                let cy = child_host.grant_yielder();
                let progs_len = child_compiled.progs.len();
                let cm = dom.source.push(child_compiled);
                let child_table = build_table_for(progs_len, 0, cm as u32);
                let cunit = dom.source.get(cm);
                let mut child_vm = match cunit
                    .and_then(|u| Vm::new(&u, entry as usize, &[Value::I64(cy as i64)]).ok())
                {
                    Some(v) => v,
                    None => {
                        vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                };
                child_vm.module = cm;
                // The coroutine lives in this vCPU's coroutine set, driven inline by `resume`.
                vt.coroutines.push(Some(Coro {
                    vm: child_vm,
                    mem: child_mem,
                    host: child_host,
                    table: child_table,
                    awaiting: None,
                    fault_yields: demand,
                    faulted_page: None,
                }));
                let h = (vt.coroutines.len() - 1) as i32;
                vt.active.set(dst, Reg::from_i32(h));
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
    /// §3.6 serve-loop core (I36 slice 1): the in-flight handler's completion ticket — `Some`
    /// between admitting a handler activation (whose return linkage rewinds into the `SvcPoll`
    /// op) and the re-execution that settles its result — and the count of dispatches completed
    /// by the current `svc.poll` activation.
    serve_ticket: Option<u64>,
    serve_count: i64,
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
            serve_ticket: None,
            serve_count: 0,
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
                self.scratch.clear();
                for &(s, _) in $copies.iter() {
                    self.scratch.push(self.regs[base + s as usize]);
                }
                for (k, &(_, d)) in $copies.iter().enumerate() {
                    self.regs[base + d as usize] = self.scratch[k];
                }
            }};
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
            step(fuel, None)?;
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
                Op::PtrAdd { dst, a, b } => {
                    r!(*dst) = Reg::from_i64(r!(*a).i64().wrapping_add(r!(*b).i64()));
                    pc += 1;
                }
                Op::PtrCast { dst, a } => {
                    r!(*dst) = Reg::from_i64(r!(*a).i64());
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
                        edge!(then_copies);
                        pc = *then_pc as usize;
                    } else {
                        edge!(else_copies);
                        pc = *else_pc as usize;
                    }
                }
                Op::BrTable { idx, arms, default } => {
                    let i = r!(*idx).i32() as u32 as usize;
                    let (copies, target) = arms.get(i).unwrap_or(default);
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
                    let callee = *callee as usize;
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
                        let argv: Box<[i64]> = args.iter().map(|a| r!(*a).i64()).collect();
                        let results: Box<[ValType]> = c.result_types[callee].clone().into();
                        // Spill past the call with the caller's window intact (no callee frame pushed);
                        // `deliver_tierup` writes the results into `dst` relative to this base.
                        self.module = module;
                        self.cur = cur;
                        self.base = base;
                        self.pc = pc + 1;
                        return Ok(Outcome::TierUp {
                            func: callee as u32,
                            argv,
                            dst: *dst as usize,
                            results,
                        });
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
                    let callee = *callee as usize;
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
                    // §3.6 (I36 slice 2) — caller-side parking: a call through a live-callee
                    // offer never reaches the generic dispatch. It enqueues on the callee's
                    // inbound queue and parks this task until the handler's reply (the
                    // tree-walker's caller-parking arm, task-level). A full callee queue is
                    // probeable backpressure (`EAGAIN` as the call's result), never a trap.
                    if let Some((callee, export)) = host.with(|p| p.live_impl_of(h, *type_id)) {
                        let t = callee
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .svc_enqueue(export, *op, argv);
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
                    let res = host.with(|p| p.cap_dispatch_slots(*type_id, *op, h, &argv, gm))?;
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
                Op::CapSelfCount { dst } => {
                    // §7 reflection op 0 — same `self_dispatch` the tree-walker uses; one i32 result.
                    let res = host.with(|p| p.self_dispatch(0, &[]))?;
                    r!(*dst) = Reg::from_i32(res[0] as i32);
                    pc += 1;
                }
                Op::CapSelfAttest { dst } => {
                    // §6 attestation op 4 — same `self_dispatch` the tree-walker / JIT thunk use.
                    let res = host.with(|p| p.self_dispatch(4, &[]))?;
                    r!(*dst) = Reg::from_i32(res[0] as i32);
                    pc += 1;
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
                Op::CapSelfGet { idx, dst } => {
                    // §7 reflection op 1 — the idx-th held cap as (handle, type_id), two i32 results.
                    let i = r!(*idx).i32() as i64;
                    let res = host.with(|p| p.self_dispatch(1, &[i]))?;
                    self.regs[base + *dst as usize] = Reg::from_i32(res[0] as i32);
                    self.regs[base + *dst as usize + 1] = Reg::from_i32(res[1] as i32);
                    pc += 1;
                }
                Op::CapSelfResolve {
                    name_ptr,
                    name_len,
                    dst,
                } => {
                    // §7 reflection op 2 — resolve a name to its handle. Through the generic dispatch
                    // (which has the window to read the name), identical to the tree-walker / JIT.
                    let ptr = r!(*name_ptr).i64();
                    let len = r!(*name_len).i64();
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let res = host.with(|p| {
                        p.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, 2, 0, &[ptr, len], gm)
                    })?;
                    r!(*dst) = Reg::from_i32(res[0] as i32);
                    pc += 1;
                }
                Op::CapSelfLabel {
                    handle,
                    buf_ptr,
                    buf_cap,
                    dst,
                } => {
                    // §7 reflection op 3 — write the handle's label into the window. Through the
                    // generic dispatch (which has the window), identical to the tree-walker / JIT.
                    let h = r!(*handle).i32() as i64;
                    let ptr = r!(*buf_ptr).i64();
                    let cap = r!(*buf_cap).i64();
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let res = host.with(|p| {
                        p.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, 3, 0, &[h, ptr, cap], gm)
                    })?;
                    r!(*dst) = Reg::from_i32(res[0] as i32);
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
                Op::ContResume { k, arg, dst } => {
                    let kh = r!(*k).i32();
                    let arg = r!(*arg).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContResume { kh, arg, dst });
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
                    return Ok(Outcome::ThreadSpawn { func, sp, arg, dst });
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
                Op::Instantiate {
                    handle,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
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
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let mh = r!(*module_reg).i64() as i32;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
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
                // §14 coroutine ops — the cap authority is resolved here (a forged/ungranted handle
                // is an inert CapFault in place), then the switch is handed to the driver.
                Op::SpawnCoroutine {
                    handle,
                    entry,
                    off,
                    size_log2,
                    dst,
                    demand,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let (dst, demand) = (*dst, *demand);
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::SpawnCoroutine {
                        ibase,
                        isize: isz,
                        entry,
                        off,
                        size_log2,
                        dst,
                        demand,
                    });
                }
                Op::SpawnCoroutineModule {
                    handle,
                    module: module_reg,
                    entry,
                    off,
                    size_log2,
                    dst,
                    demand,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let mh = r!(*module_reg).i64() as i32;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let (dst, demand) = (*dst, *demand);
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::SpawnCoroutineModule {
                        ibase,
                        isize: isz,
                        mh,
                        entry,
                        off,
                        size_log2,
                        dst,
                        demand,
                    });
                }
                Op::CoResume {
                    handle,
                    ch,
                    value,
                    dst,
                } => {
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?; // authority
                    let ch = r!(*ch).i32();
                    let value = r!(*value).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::CoResume { ch, value, dst });
                }
                Op::CoYield { handle, value, dst } => {
                    let yh = r!(*handle).i32();
                    host.with(|p| p.resolve_yielder(yh))?; // authority
                    let value = r!(*value).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::CoYield { value, dst });
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
            }
        }
    }
}
