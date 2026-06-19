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
    BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx, IToF, Inst,
    IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValType,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, slot_to_val, step, trunc_trap, val_to_slot, GuestMem, Host, Mem, Reg, Trap,
    Value, DEFAULT_RESERVED_LOG2,
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
    /// §7 reflection `cap.self.count` — number of caps this domain holds (one `i32` result).
    CapSelfCount {
        dst: u32,
    },
    /// §7 reflection `cap.self.get` — the `idx`-th held cap as `(handle, type_id)` (two `i32`
    /// results in `dst`, `dst+1`).
    CapSelfGet {
        idx: u32,
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

/// One slot of a domain's `call_indirect` dispatch table: `(module, func)`, where module 0 is the
/// primary program and `k ≥ 1` is `mods[k]` (an installed §22 unit). `None` is a trapping padding
/// slot. The flat analogue of the tree-walker's [`crate::DomainTable`] (`module<<32 | func`).
type Table = Vec<Option<(u32, u32)>>;

/// Build a dispatch table of `2^table_log2` (at least `next_power_of_two(n_funcs)`) slots: the first
/// `n_funcs` map to `(module 0, i)`, the rest are trapping padding (fillable by `install`).
fn build_table(n_funcs: usize, table_log2: u8) -> Table {
    build_table_for(n_funcs, table_log2, 0)
}

/// [`build_table`] but for an arbitrary `module` index — the natural table a §14 separate-**module**
/// child uses, whose funcs live at `dom.mods[module]` (not module 0).
fn build_table_for(n_funcs: usize, table_log2: u8, module: u32) -> Table {
    let len = (1usize << table_log2)
        .max(n_funcs.next_power_of_two())
        .max(1);
    (0..len)
        .map(|i| (i < n_funcs).then_some((module, i as u32)))
        .collect()
}

/// A running domain: its compiled modules (`mods[0]` = primary, `mods[k≥1]` = installed §22 units)
/// plus the shared `call_indirect` dispatch table. `install` appends a unit and fills a padding slot;
/// `uninstall` clears one. Owned by [`drive`]; vCPUs read it, and `install`/`uninstall` mutate it
/// between op steps (cooperative single-thread, so no atomics — unlike the tree-walker's `DomainTable`).
struct Domain {
    mods: Vec<Compiled>,
    table: Table,
}

impl Domain {
    fn new(primary: Compiled, table_log2: u8) -> Domain {
        let table = build_table(primary.progs.len(), table_log2);
        Domain {
            mods: vec![primary],
            table,
        }
    }

    /// `Jit.install`: append `unit` (its module id = `mods.len()`), fill the first padding slot with
    /// `(module, 0)`, and return that slot — or `None` if the table is full (`-ENOSPC`).
    fn install(&mut self, unit: Compiled) -> Option<usize> {
        let slot = self.table.iter().position(|e| e.is_none())?;
        let module = self.mods.len() as u32;
        self.mods.push(unit);
        self.table[slot] = Some((module, 0));
        Some(slot)
    }

    /// `Jit.uninstall`: clear a filled padding slot (`≥ n_real`) back to trapping, returning success.
    /// A real-function slot (`< n_real`), out-of-range, or already-empty slot is rejected.
    fn uninstall(&mut self, slot: usize, n_real: usize) -> bool {
        if slot >= n_real && slot < self.table.len() && self.table[slot].is_some() {
            self.table[slot] = None;
            true
        } else {
            false
        }
    }
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
    let mut has_coro = false;
    let mut has_fiber = false;
    let mut has_thread = false;
    let mut has_instantiate = false;
    let mut has_gc = false;
    for f in funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    // ops 0/1 = instantiate/join, op 5 = instantiate_module (executor children);
                    // everything else on INSTANTIATOR/YIELDER is the inline coroutine round-trip.
                    Inst::CapCall {
                        type_id: super::iface::INSTANTIATOR,
                        op: 0 | 1 | 5,
                        ..
                    } => has_instantiate = true,
                    Inst::CapCall {
                        type_id: super::iface::INSTANTIATOR | super::iface::YIELDER,
                        ..
                    } => has_coro = true,
                    Inst::ContNew { .. } | Inst::ContResume { .. } | Inst::Suspend { .. } => {
                        has_fiber = true
                    }
                    Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. } => has_thread = true,
                    Inst::GcRoots { .. } => has_gc = true,
                    _ => {}
                }
            }
        }
    }
    // `gc.roots` scans only the **calling vCPU's** continuation (its stack, fibers, coroutines), so a
    // module that also spawns threads could hold roots in a sibling vCPU we wouldn't scan — reject
    // that combination (fall back) to stay sound. `gc.roots` + fibers / coroutines is fine (those
    // continuations *are* scanned).
    if (has_coro && (has_fiber || has_thread))
        || (has_instantiate && has_fiber)
        || (has_gc && has_thread)
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
            use super::iface;
            match (*type_id, *op) {
                // §14 executor children — instantiate (op 0) spawns a confined child on the scheduler;
                // join (op 1) parks until it finishes, reusing the §12 thread join machinery (children
                // share the `threads` handle namespace). The separate-module / demand variants (5/6/7
                // and op 4) and the JIT / SharedRegion-grant variants need seams this slice doesn't
                // drive: reject (fall back).
                (iface::INSTANTIATOR, 0) if args.len() >= 4 => Op::Instantiate {
                    handle: g(*handle),
                    entry: g(args[0]),
                    off: g(args[1]),
                    size_log2: g(args[2]),
                    quota: g(args[3]),
                    dst,
                },
                (iface::INSTANTIATOR, 1) if !args.is_empty() => Op::InstJoin {
                    handle: g(*handle),
                    child: g(args[0]),
                    dst,
                },
                // op 5 = instantiate_module: the first arg is the granted `Module` handle; the carve
                // args (entry/off/size_log2/quota) follow. (join, op 1, serves both kinds.)
                (iface::INSTANTIATOR, 5) if args.len() >= 5 => Op::InstantiateModule {
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
                (iface::INSTANTIATOR, op @ (6 | 7)) if args.len() >= 4 => {
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
                // §14 cooperative coroutine round-trip — spawn_coroutine (op 2) / spawn_demand_coroutine
                // (op 4, window starts unmapped) / resume / yield.
                (iface::INSTANTIATOR, op @ (2 | 4)) if args.len() >= 3 => Op::SpawnCoroutine {
                    handle: g(*handle),
                    entry: g(args[0]),
                    off: g(args[1]),
                    size_log2: g(args[2]),
                    dst,
                    demand: op == 4,
                },
                (iface::INSTANTIATOR, 3) if args.len() >= 2 => Op::CoResume {
                    handle: g(*handle),
                    ch: g(args[0]),
                    value: g(args[1]),
                    dst,
                },
                (iface::YIELDER, 0) if !args.is_empty() => Op::CoYield {
                    handle: g(*handle),
                    value: g(args[0]),
                    dst,
                },
                // §22 guest-driven JIT units: install/uninstall drive the dispatch table; compile /
                // compile_linked (ops 0/5) are pure host ops, so they fall through to the generic
                // dispatch below. `invoke` (op 1) is the next slice — reject it for now (fall back).
                (iface::JIT, 3) if !args.is_empty() => Op::JitInstall {
                    handle: g(*handle),
                    code: g(args[0]),
                    dst,
                },
                (iface::JIT, 4) if !args.is_empty() => Op::JitUninstall {
                    handle: g(*handle),
                    slot: g(args[0]),
                    dst,
                },
                (iface::JIT, 1) if !args.is_empty() => Op::JitInvoke {
                    handle: g(*handle),
                    code: g(args[0]),
                    args: args[1..].iter().map(|a| g(*a)).collect(),
                    dst,
                    // The cap.call sig is `(i64 code, params…) -> (results…)`; the unit entry's
                    // params are sig.params without the leading code-handle.
                    params: sig.params.get(1..).unwrap_or(&[]).to_vec().into(),
                    results: sig.results.clone().into(),
                },
                (iface::INSTANTIATOR, _) | (iface::YIELDER, _) => return None,
                (iface::SHARED_REGION, 4) => return None,
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
        Inst::CapSelfGet { idx } => Op::CapSelfGet { idx: g(*idx), dst },
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
        Inst::CallImport { .. } => return None,
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

/// A run result paired with its trap-time backtrace (innermost frame first, as [`crate::IrPc`]s;
/// empty on a clean finish) — what [`compile_and_run_with_host_traced`] returns.
pub type TracedRun = (Result<Vec<Value>, Trap>, Vec<super::IrPc>);

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
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = build_mem(m);
    let mut vm = match Vm::new(&dom.mods[0], func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Err(e), Vec::new())),
    };
    loop {
        match vm.resume(&dom.mods, &dom.table, fuel, &mut mem, host, 1) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((Ok(vals), Vec::new())),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope (fall back to tree-walker)
            Err(t) => {
                let bt = vm_trap_bt(&vm, &dom.mods, &t);
                return Some((Err(t), bt));
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
fn vm_trap_bt(vm: &Vm, mods: &[Compiled], trap: &Trap) -> Vec<super::IrPc> {
    let mut bt = Vec::new();
    if let Some((block, inst)) = mods[vm.module].progs[vm.cur].src.get(vm.pc).copied().flatten() {
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
        if let Some((block, inst)) = mods[module].progs[prog].src.get(call_pc).copied().flatten() {
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
    // tree-walker), lest it write a silently-wrong artifact.
    let threadish = m.funcs.iter().flat_map(|f| f.blocks.iter()).any(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i, Inst::ThreadSpawn { .. } | Inst::ThreadJoin { .. }))
    });
    if threadish {
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
    let mods = [c];
    let table = build_table(mods[0].progs.len(), 0);
    let mut mem = build_mem(m);
    let mut host = Host::new();
    let mut vm = match Vm::new(&mods[0], func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        if let Some(pc) = vm.cur_ir_pc(&mods) {
            trace.push(pc);
        }
        match vm.resume(&mods, &table, fuel, &mut mem, &mut host, 1) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
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
    let sp = m.durable_get_sp();
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
    m.durable_set_sp(in_sp);
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
    let root_sp = ctx
        .mem
        .as_ref()
        .map(|m| m.durable_get_sp())
        .unwrap_or(super::SHADOW_BASE);
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
            m.durable_set_sp(super::shadow_region_base(slot + 1));
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
            .map(|m| m.durable_get_sp())
            .unwrap_or(super::SHADOW_BASE);
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
        m.durable_set_sp(root_sp);
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
    table: Table,
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
fn resume_coro(coro: &mut Coro, mods: &[Compiled], fuel: &mut u64) -> Result<CoStop, Trap> {
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
            mods,
            &coro.table,
            fuel,
            &mut coro.mem,
            &mut coro.host,
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
                    scan_vm_roots(&coro.vm, mods, &mut consider);
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
fn scan_vm_roots(vm: &Vm, mods: &[Compiled], consider: &mut impl FnMut(u64)) {
    let frames = std::iter::once((vm.module, vm.cur, vm.base))
        .chain(vm.stack.iter().map(|&(m, p, b, _, _)| (m, p, b)));
    for (module, prog, base) in frames {
        let n = mods[module].progs[prog].nslots as usize;
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
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    let mut vm = Vm::new(&dom.mods[module], 0, args)?;
    vm.module = module;
    loop {
        match vm.resume(&dom.mods, &dom.table, fuel, mem, host, u64::MAX)? {
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
    Done(Vec<Value>),
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
}

/// The per-vCPU execution environment a [`step_vcpu`] runs against: the dispatch `table` it uses
/// (the shared domain table, or a §14 confined child's own natural table), its `fuel` budget, its
/// linear `mem`, and its capability `host`. The root vCPU and its `thread.spawn` siblings share the
/// domain's (env `None`); a §14 `instantiate` child carries its own confined [`ChildEnv`]. Bundled so
/// [`step_vcpu`] takes one ref instead of four (and so the per-task selection has a single type).
struct RunCtx<'a> {
    table: &'a Table,
    fuel: &'a mut u64,
    mem: &'a mut Option<Mem>,
    host: &'a mut Host,
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
            &dom.mods,
            ctx.table,
            &mut *ctx.fuel,
            &mut *ctx.mem,
            &mut *ctx.host,
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
                fiber_sp.push(super::shadow_region_base(h as usize + 1));
                // Freeze residue (DURABILITY.md §12.8): record the fiber's re-entry metadata — its
                // **resolved** entry function index (the natural-table lookup `cont.resume` does, so
                // a `FrozenFiber.func` matches the tree-walker's `Frame::func`) and data-stack base —
                // so the freeze driver can emit a `FrozenFiber` for it even after it parks.
                let func_idx = (funcref as u32 as usize & dom.mods[0].table_mask) as i32;
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
                        let m0 = &dom.mods[0];
                        let f = (funcref as u32 as usize) & m0.table_mask;
                        let ok = m0
                            .sigs
                            .get(f)
                            .is_some_and(|(p, r)| p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS);
                        if !ok {
                            return Err(Trap::FiberFault);
                        }
                        Vm::new(m0, f, &[Value::I64(sp), Value::I64(arg)])?
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
            Outcome::ThreadSpawn { func, sp, arg, dst } => {
                return Ok(VcpuStop::Spawn { func, sp, arg, dst })
            }
            Outcome::ThreadJoin { handle, dst } => return Ok(VcpuStop::Join { handle, dst }),
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
                    &dom.mods[0],
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
                match resume_coro(&mut coro, &dom.mods, &mut *ctx.fuel)? {
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
                    scan_vm_roots(&vt.active, &dom.mods, &mut consider);
                    for (_, vm, _) in &vt.chain {
                        scan_vm_roots(vm, &dom.mods, &mut consider);
                    }
                    for fib in fibers.iter() {
                        if let FiberState::Parked { vm, .. } = fib {
                            scan_vm_roots(vm, &dom.mods, &mut consider);
                        }
                    }
                    for coro in vt.coroutines.iter().flatten() {
                        scan_vm_roots(&coro.vm, &dom.mods, &mut consider);
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
    host: Host,
    table: Table,
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
    mut dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    budget: u64,
) -> Result<Vec<Value>, Trap> {
    let mut tasks: Vec<TaskSlot> = vec![TaskSlot {
        vt: VTask::new(&dom.mods[0], entry as usize, args)?,
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
                    host: &mut *host,
                };
                match freeze_drive(
                    &mut fibers,
                    &mut fiber_sp,
                    &mut fiber_meta,
                    &dom,
                    &mut ctx,
                    budget,
                ) {
                    Ok(frozen) => host.frozen_fibers = frozen,
                    Err(t) => return Err(t),
                }
            }
            return res;
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
                host: &mut *host,
            },
            Some(k) => {
                let e = &mut extra_envs[k];
                RunCtx {
                    table: &e.table,
                    fuel: &mut e.fuel,
                    mem: &mut e.mem,
                    durable: e.host.is_durable(),
                    host: &mut e.host,
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
            Ok(VcpuStop::Spawn { func, sp, arg, dst }) => {
                if func as usize >= dom.mods[0].progs.len() {
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
                    &dom.mods[0],
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
                let c0 = &dom.mods[0];
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
                let c0 = &dom.mods[0];
                let child_table = build_table(c0.progs.len(), 0);
                let child_vt = VTask::new(c0, entry as usize, &child_args)?;
                let eidx = extra_envs.len();
                extra_envs.push(ChildEnv {
                    mem: child_mem,
                    host: child_host,
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
                let (cfuncs, cmem_log2, cdata) = match host.resolve_module(mh) {
                    Ok(g) => (g.funcs.clone(), g.memory_log2, g.data.clone()),
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
                let cm = dom.mods.len();
                dom.mods.push(child_compiled);
                let child_table = build_table_for(progs_len, 0, cm as u32);
                let mut child_vt = VTask::new(&dom.mods[cm], entry as usize, &child_args)?;
                child_vt.active.module = cm;
                let eidx = extra_envs.len();
                extra_envs.push(ChildEnv {
                    mem: child_mem,
                    host: child_host,
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
                let cm = dom.mods.len();
                dom.mods.push(child_compiled);
                let child_table = build_table_for(progs_len, 0, cm as u32);
                let mut child_vm =
                    match Vm::new(&dom.mods[cm], entry as usize, &[Value::I64(cy as i64)]) {
                        Ok(v) => v,
                        Err(_) => {
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
                let n_real = dom.mods[0].progs.len();
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
                let umod = dom.mods.len();
                dom.mods.push(unit);
                match run_invoke(&dom, umod, &child_args, fuel, mem, host) {
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
    fn cur_ir_pc(&self, mods: &[Compiled]) -> Option<super::IrPc> {
        let (block, inst) = mods[self.module].progs[self.cur]
            .src
            .get(self.pc)
            .copied()
            .flatten()?;
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
        mods: &[Compiled],
        table: &Table,
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut Host,
        mut budget: u64,
    ) -> Result<Outcome, Trap> {
        let mut module = self.module;
        let mut cur = self.cur;
        let mut base = self.base;
        let mut pc = self.pc;
        // The current module's compiled program, re-bound only when an activation crosses modules
        // (cross-module `call_indirect` / its return) — so the per-op hot path is unchanged.
        let mut c: &Compiled = &mods[module];

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
            step(fuel)?;
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
                Op::Call { callee, args, dst } => {
                    let callee = *callee as usize;
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
                    let (tmod, tfunc) = match table[slot] {
                        Some(e) => (e.0 as usize, e.1 as usize),
                        None => return Err(Trap::IndirectCallType),
                    };
                    let (cp, cr) = &mods[tmod].sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + mods[tmod].progs[tfunc].nslots as usize;
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
                        c = &mods[tmod];
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
                            c = &mods[cmod];
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
                    let (tmod, tfunc) = match table[slot] {
                        Some(e) => (e.0 as usize, e.1 as usize),
                        None => return Err(Trap::IndirectCallType),
                    };
                    let (cp, cr) = &mods[tmod].sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let need = base + mods[tmod].progs[tfunc].nslots as usize;
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
                        c = &mods[tmod];
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
                    // as i64 slots, results re-typed by the call's `sig.results`. The host is borrowed
                    // exclusively here (single-threaded, no `thread.spawn` in a compiled module), so no
                    // lock is needed.
                    let h = r!(*handle).i32();
                    let mut argv: Vec<i64> = Vec::with_capacity(args.len());
                    for a in args.iter() {
                        argv.push(r!(*a).i64());
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let res = host.cap_dispatch_slots(*type_id, *op, h, &argv, gm)?;
                    for (i, (s, ty)) in res.iter().zip(results.iter()).enumerate() {
                        self.regs[base + *dst as usize + i] = Reg::from_value(slot_to_val(*ty, *s));
                    }
                    pc += 1;
                }
                Op::CapSelfCount { dst } => {
                    // §7 reflection op 0 — same `self_dispatch` the tree-walker uses; one i32 result.
                    let res = host.self_dispatch(0, &[])?;
                    r!(*dst) = Reg::from_i32(res[0] as i32);
                    pc += 1;
                }
                Op::CapSelfGet { idx, dst } => {
                    // §7 reflection op 1 — the idx-th held cap as (handle, type_id), two i32 results.
                    let i = r!(*idx).i32() as i64;
                    let res = host.self_dispatch(1, &[i])?;
                    self.regs[base + *dst as usize] = Reg::from_i32(res[0] as i32);
                    self.regs[base + *dst as usize + 1] = Reg::from_i32(res[1] as i32);
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
                Op::Instantiate {
                    handle,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                } => {
                    let (ibase, isz) = host.resolve_instantiator(r!(*handle).i32())?;
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
                    let (ibase, isz) = host.resolve_instantiator(r!(*handle).i32())?;
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
                    host.resolve_instantiator(r!(*handle).i32())?; // authority
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
                    let (ibase, isz) = host.resolve_instantiator(r!(*handle).i32())?;
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
                    let (ibase, isz) = host.resolve_instantiator(r!(*handle).i32())?;
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
                    host.resolve_instantiator(r!(*handle).i32())?; // authority
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
                    host.resolve_yielder(r!(*handle).i32())?; // authority
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
            }
        }
    }
}
