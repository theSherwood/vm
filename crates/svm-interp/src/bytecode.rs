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
//! Scope (this slice): scalar + memory + SIMD/`v128` + fences + direct calls. Hot scalar/memory ops
//! dispatch inline; the SIMD/`v128`/fence long tail is delegated to the reference
//! [`super::eval_inst`] (same semantics, no re-implementation). [`compile_module`] returns `None`
//! when a function uses an op that needs a runtime seam this slice doesn't drive — `call_indirect`
//! (dispatch table), tail calls, or capability/fiber/thread/reflection ops (host powerbox,
//! scheduler, fiber registry) — so callers fall back to the tree-walker. It is **not yet wired as
//! the default execution path**; correctness is gated by the equality harness
//! (`crates/svm/tests/bytecode_diff.rs`) against the reference interpreter, and the remaining seams
//! (debug, scheduler, fibers, durability, capabilities) plus those ops land in later slices.
//!
//! Like the reference interpreter, it is total and panic-free: every slot/pc index is in range by
//! construction of the compiler, and `compile_module` rejects anything it can't lower.

use svm_ir::{
    BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx, IToF, Inst,
    IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValType,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, step, trunc_trap, Mem, Reg, Trap, Value, DEFAULT_RESERVED_LOG2,
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
        // Control / host / cross-module ops this slice doesn't drive (they need the scheduler,
        // host powerbox, fiber registry, or dispatch table) — fall back to the tree-walker.
        Inst::CapCall { .. }
        | Inst::CapSelfCount
        | Inst::CapSelfGet { .. }
        | Inst::CallImport { .. }
        | Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. }
        | Inst::GcRoots { .. } => return None,
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
    let c = compile_module(&m.funcs)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let mut mem = build_mem(m);
    Some(run(&c, func, args, fuel, &mut mem))
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
    let mut vm = match Vm::new(&c, func as usize, args) {
        Ok(v) => v,
        Err(t) => return Some(Err(t)),
    };
    loop {
        match vm.resume(&c, fuel, &mut mem, slice.max(1)) {
            Ok(Outcome::Done(vals)) => return Some(Ok(vals)),
            Ok(Outcome::Suspended) => continue,
            Err(t) => return Some(Err(t)),
        }
    }
}

fn run(
    c: &Compiled,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
) -> Result<Vec<Value>, Trap> {
    let mut vm = Vm::new(c, entry as usize, args)?;
    // The production path never preempts itself: an unlimited budget makes `resume` run straight to
    // completion, with the per-op budget branch perfectly predicted (so the hot loop is unchanged).
    loop {
        match vm.resume(c, fuel, mem, u64::MAX)? {
            Outcome::Done(vals) => return Ok(vals),
            Outcome::Suspended => continue,
        }
    }
}

/// Why [`Vm::resume`] returned: the entry activation completed (`Done`), or it hit its op budget at
/// an op boundary with the cursor persisted into the `Vm` (`Suspended`) — call `resume` again to
/// continue. (A trap is the `Err` arm of `resume`'s `Result` and is terminal, like the tree-walker.)
enum Outcome {
    Done(Vec<Value>),
    Suspended,
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
