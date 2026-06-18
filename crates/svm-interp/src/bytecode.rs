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
//! ([`drive`]) over one shared `Mem` — faithful for the interleaving-invariant programs the oracle
//! uses. Hot scalar/memory ops dispatch inline; the SIMD/`v128`/fence long tail is delegated to the
//! reference [`super::eval_inst`] (same semantics, no re-implementation). [`compile_module`] returns
//! `None` when a function needs a seam not yet driven here — `Instantiator`/`Yielder` coroutines,
//! cross-module `install`/`invoke`, tail calls, durability, **or** a module mixing threads *and*
//! fibers (migration needs a run-shared registry) — so callers (`super::run_with_host_fast`) fall
//! back to the tree-walker for those.
//!
//! `run`/`run_with_host` stay the tree-walker (the reference oracle); the bytecode engine is reached
//! via `run_fast`/`run_with_host_fast`. Correctness is gated by exact-equality harnesses against the
//! tree-walker (`bytecode_diff.rs`, `bytecode_caps.rs`, `bytecode_fibers.rs`, `bytecode_threads.rs`).
//!
//! Like the reference interpreter, it is total and panic-free: every slot/pc index is in range by
//! construction of the compiler, and `compile_module` rejects anything it can't lower.

use svm_ir::{
    BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx, IToF, Inst,
    IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValType,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, slot_to_val, step, trunc_trap, GuestMem, Host, Mem, Reg, Trap, Value,
    DEFAULT_RESERVED_LOG2,
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
    Ret {
        srcs: Box<[u32]>,
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

struct Program {
    ops: Vec<Op>,
    nslots: u32,
}

/// A whole compiled module: one [`Program`] per function plus each function's result types (for
/// reconstructing typed `Value`s at the entry boundary).
pub struct Compiled {
    progs: Vec<Program>,
    result_types: Vec<Vec<ValType>>,
    /// Per-function `(params, results)` for `call_indirect` type-checking — the natural module-0
    /// function table indexes these directly (slot `i` ⇒ func `i`).
    sigs: Vec<(Vec<ValType>, Vec<ValType>)>,
    /// `len - 1` of the natural table (`next_power_of_two(n_funcs)`), the `call_indirect` slot mask.
    table_mask: usize,
}

/// Lower every function, or `None` if any uses an op outside this slice's subset.
pub fn compile_module(funcs: &[Func]) -> Option<Compiled> {
    // The bytecode engine drives threads with a **per-vCPU** fiber registry, whereas the tree-walker
    // shares one fiber-handle namespace across a domain's vCPUs (so a fiber can migrate, and handles
    // are domain-global). A module that uses *both* threads and fibers would therefore diverge on
    // handle numbering / migration, so reject the combination here (→ tree-walker fallback) until the
    // run-shared registry lands (Slice 1c-5c migration follow-up).
    let mut has_threads = false;
    let mut has_fibers = false;
    for f in funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    Inst::ThreadSpawn { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. } => has_threads = true,
                    Inst::ThreadJoin { .. } => has_threads = true,
                    Inst::ContNew { .. } | Inst::ContResume { .. } | Inst::Suspend { .. } => {
                        has_fibers = true
                    }
                    _ => {}
                }
            }
        }
    }
    if has_threads && has_fibers {
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
            // Tail calls / indirect tail calls: not in this slice.
            _ => return None,
        }
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
    Some(Program { ops, nslots })
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
            if matches!(*type_id, iface::INSTANTIATOR | iface::YIELDER | iface::JIT)
                || (*type_id == iface::SHARED_REGION && *op == 4)
            {
                return None;
            }
            Op::CapCall {
                type_id: *type_id,
                op: *op,
                handle: g(*handle),
                args: args.iter().map(|a| g(*a)).collect(),
                dst,
                results: sig.results.clone().into(),
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
        Inst::CallImport { .. } | Inst::GcRoots { .. } => return None,
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
    let mut mem = build_mem(m);
    Some(run(&c, func, args, fuel, &mut mem, host))
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
    let mut mem = build_mem(m);
    let mut host = Host::new();
    Some(drive(
        &c,
        func,
        args,
        fuel,
        &mut mem,
        &mut host,
        slice.max(1),
    ))
}

fn run(
    c: &Compiled,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    // The production path never preempts itself: an unlimited budget makes `resume` run straight to
    // completion, with the per-op budget branch perfectly predicted (so the hot loop is unchanged).
    drive(c, entry, args, fuel, mem, host, u64::MAX)
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
}

/// A §12 fiber's state in the driver's per-vCPU registry (handle = index). Non-durable only — a
/// durable run falls back to the tree-walker (`run_with_host_fast` gates on `!host.is_durable()`).
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

/// One vCPU's continuation: its active `Vm` plus its §12 fiber world (per-vCPU registry + resume
/// `chain`). A `thread.spawn` creates a fresh `VTask`; the scheduler runs them cooperatively over one
/// shared `Mem` (single-threaded, so shared memory is sequentially consistent — the determinate
/// programs the oracle uses give the same result on any correct schedule).
struct VTask {
    active: Vm,
    /// `ROOT_FIBER` or the handle of the fiber currently running in this vCPU.
    active_id: usize,
    /// Parked resumers: `(fiber id, its Vm, the `cont.resume` result slot awaiting (status, value))`.
    chain: Vec<(usize, Vm, u32)>,
    /// This vCPU's fiber registry (handle = index). Per-vCPU — combined thread+fiber modules are
    /// rejected by the compiler until the run-shared registry (migration) lands in a later slice.
    fibers: Vec<FiberState>,
}

impl VTask {
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<VTask, Trap> {
        Ok(VTask {
            active: Vm::new(c, entry, args)?,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            fibers: Vec::new(),
        })
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
}

/// Run one vCPU (its active `Vm` and any fibers it switches among) until it finishes or hits a
/// multi-vCPU event. Fiber `Outcome`s are serviced here exactly as `run_inner`'s `cont.*` arms switch
/// the active frame stack; `thread.*`/`memory.*` `Outcome`s are handed up to [`drive`]. `budget` only
/// slices *where* the active `Vm` pauses (Slice 1c-2); it never changes results.
fn step_vcpu(
    vt: &mut VTask,
    c: &Compiled,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    budget: u64,
) -> Result<VcpuStop, Trap> {
    loop {
        match vt.active.resume(c, fuel, mem, host, budget)? {
            // Budget exhausted (sliced harness only): re-enter the same activation; its cursor is
            // already persisted, so this is transparent.
            Outcome::Suspended => {}
            Outcome::Done(vals) => match vt.chain.pop() {
                // The vCPU's root activation finished.
                None => return Ok(VcpuStop::Done(vals)),
                // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` to its resumer.
                Some((rid, resumer, rdst)) => {
                    vt.fibers[vt.active_id] = FiberState::Done;
                    let retval = vals.first().copied().unwrap_or(Value::I64(0));
                    vt.active = resumer;
                    vt.active_id = rid;
                    vt.active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                    vt.active.set(rdst + 1, Reg::from_value(retval));
                }
            },
            Outcome::ContNew { funcref, sp, dst } => {
                if vt.fibers.len() + 1 >= super::MAX_FIBERS {
                    return Err(Trap::FiberFault);
                }
                let h = vt.fibers.len() as i32;
                vt.fibers.push(FiberState::Pending { funcref, sp });
                vt.active.set(dst, Reg::from_i32(h));
            }
            Outcome::ContResume { kh, arg, dst } => {
                let k = kh as usize;
                // Claim fiber `k`: a pending fiber starts (call `funcref(sp, arg)`), a parked one
                // continues (the new `arg` becomes its `suspend`'s result). Anything else is inert.
                let target = match vt.fibers.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let (funcref, sp) = match std::mem::replace(slot, FiberState::Running) {
                            FiberState::Pending { funcref, sp } => (funcref, sp),
                            _ => unreachable!(),
                        };
                        // Resolve the fiber entry through the natural table + `fiber_sig`, exactly
                        // as `table_lookup` does — a forged/mistyped funcref is a `FiberFault`.
                        let f = (funcref as u32 as usize) & c.table_mask;
                        let ok = c
                            .sigs
                            .get(f)
                            .is_some_and(|(p, r)| p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS);
                        if !ok {
                            return Err(Trap::FiberFault);
                        }
                        Vm::new(c, f, &[Value::I64(sp), Value::I64(arg)])?
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
                let resumer = std::mem::replace(&mut vt.active, target);
                vt.chain.push((vt.active_id, resumer, dst));
                vt.active_id = k;
            }
            Outcome::FiberSuspend { value, dst } => {
                // Pop the resumer to switch back to; an empty chain means the root tried to
                // `suspend`, which is a `FiberFault` (the root has no resumer).
                let (rid, resumer, rdst) = vt.chain.pop().ok_or(Trap::FiberFault)?;
                let suspended = std::mem::replace(&mut vt.active, resumer);
                vt.fibers[vt.active_id] = FiberState::Parked {
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
        }
    }
}

/// `fiber_sig` params/results, inlined so the driver can compare without allocating a `FuncType`.
const FIBER_PARAMS: [ValType; 2] = [ValType::I64, ValType::I64];
const FIBER_RESULTS: [ValType; 1] = [ValType::I64];

/// A scheduled vCPU and its blocking state.
struct TaskSlot {
    vt: VTask,
    /// This vCPU's `thread.spawn` children (handle = index → global task index). `None` = joined.
    threads: Vec<Option<usize>>,
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
    c: &Compiled,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    budget: u64,
) -> Result<Vec<Value>, Trap> {
    let mut tasks: Vec<TaskSlot> = vec![TaskSlot {
        vt: VTask::new(c, entry as usize, args)?,
        threads: Vec::new(),
        state: TaskState::Runnable,
    }];
    let mut clock: u64 = 0;

    loop {
        // The root's result is the run's result (other vCPUs' effects are already reflected in it).
        if let TaskState::Done(res) = &tasks[0].state {
            return res.clone();
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

        match step_vcpu(&mut tasks[ti].vt, c, fuel, mem, host, budget) {
            Err(trap) => complete(&mut tasks, ti, Err(trap)),
            Ok(VcpuStop::Done(vals)) => complete(&mut tasks, ti, Ok(vals)),
            Ok(VcpuStop::Spawn { func, sp, arg, dst }) => {
                if func as usize >= c.progs.len() {
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
                let child = VTask::new(c, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
                let cidx = tasks.len();
                tasks.push(TaskSlot {
                    vt: child,
                    threads: Vec::new(),
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
    /// Suspended caller activations: `(prog, base, resume pc, absolute first result slot)`.
    stack: Vec<(usize, usize, usize, usize)>,
    /// The running activation's program (function index), window base, and op cursor.
    cur: usize,
    base: usize,
    pc: usize,
    /// Edge-copy staging buffer (parallel-copy safety); kept here so it is reused across resumes.
    scratch: Vec<Reg>,
}

impl Vm {
    /// Open the entry activation: a zero-based window sized to the entry function, seeded with the
    /// call arguments. Total — an out-of-range entry or arg overflow is a clean `Malformed` trap.
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<Vm, Trap> {
        let prog = c.progs.get(entry).ok_or(Trap::Malformed)?;
        let mut regs: Vec<Reg> = vec![Reg::default(); prog.nslots as usize];
        for (i, a) in args.iter().enumerate() {
            *regs.get_mut(i).ok_or(Trap::Malformed)? = Reg::from_value(*a);
        }
        Ok(Vm {
            regs,
            stack: Vec::new(),
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
        c: &Compiled,
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut Host,
        mut budget: u64,
    ) -> Result<Outcome, Trap> {
        let mut cur = self.cur;
        let mut base = self.base;
        let mut pc = self.pc;

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
                    r!(*dst) = Reg::from_value(m.load(a, *offset, *op)?);
                    pc += 1;
                }
                Op::Store {
                    addr,
                    value,
                    op,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let v = Value::I64(r!(*value).i64());
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .store(a, *offset, *op, v)?;
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
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack.push((cur, base, pc + 1, base + *dst as usize));
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
                    // Resolve through the natural module-0 table (slot i ⇒ func i), then type-check
                    // the resolved signature — a forged/mistyped slot is an inert IndirectCallType
                    // trap.
                    let slot = (r!(*idx).i32() as u32 as usize) & c.table_mask;
                    let callee = if slot < c.sigs.len() {
                        slot
                    } else {
                        return Err(Trap::IndirectCallType);
                    };
                    let (cp, cr) = &c.sigs[callee];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack.push((cur, base, pc + 1, base + *dst as usize));
                    cur = callee;
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
                    Some((cprog, cbase, cpc, ret_abs)) => {
                        for (i, s) in srcs.iter().enumerate() {
                            self.regs[ret_abs + i] = self.regs[base + *s as usize];
                        }
                        cur = cprog;
                        base = cbase;
                        pc = cpc;
                    }
                },
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
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContResume { kh, arg, dst });
                }
                Op::Suspend { value, dst } => {
                    let value = r!(*value).i64();
                    let dst = *dst;
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
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ThreadSpawn { func, sp, arg, dst });
                }
                Op::ThreadJoin { handle, dst } => {
                    let handle = r!(*handle).i32();
                    let dst = *dst;
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
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::MemoryNotify {
                        base: base_addr,
                        count,
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
