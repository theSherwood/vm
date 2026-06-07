//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use svm_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, Data, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func,
    FuncIdx, FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValIdx,
    ValType, DEFAULT_RESERVED_LOG2,
};
use svm_mask::Window;

/// A runtime value. Mirrors `ValType`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Reasons execution stopped without producing results.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Trap {
    /// Ran out of fuel (potential infinite loop) — see `run`.
    OutOfFuel,
    /// Integer division or remainder by zero (§3b).
    DivByZero,
    /// Signed `div_s` of `INT_MIN / -1`: the quotient `+2^31` is not representable, so
    /// it traps (§3b: trap only when there is no representable result). `rem_s` does
    /// **not** trap here — the remainder `0` *is* representable.
    IntOverflow,
    /// A memory access crossed the top of the window (guard-region fault, §4/§5).
    MemoryFault,
    /// Call recursion exceeded the interpreter's depth bound (host-stack guard).
    StackOverflow,
    /// `call_indirect` selected an empty table slot or a function whose signature
    /// did not match the call's type (the §3c table type-id check).
    IndirectCallType,
    /// Reached an `unreachable`/`trap` terminator (§3b).
    Unreachable,
    /// A trapping float→int conversion saw NaN or an out-of-range value (§3b).
    BadConversion,
    /// A `cap.call` named a handle that is forged, closed/revoked (dead generation),
    /// or the wrong interface type — the index was **inert** (§3c). Not an escape.
    CapFault,
    /// The guest invoked the `Exit` capability; carries the requested exit code. Not
    /// an error — the domain asked to terminate (§3e). Propagates like a trap.
    Exit(i32),
    /// Structurally invalid in a way a verified module never is (defensive only).
    Malformed,
}

/// Maximum nested `call` depth before the interpreter traps, bounding the size of the
/// **explicit** guest call stack (a `Vec<Frame>`, §12) so adversarial (or merely deep)
/// guest recursion yields a clean `Trap::StackOverflow` rather than unbounded growth.
///
/// The interpreter no longer recurses on the host stack — the guest call stack is
/// reified (so a fiber's continuation is just its `Vec<Frame>`, suspendable; §12), and
/// the host stack stays O(1) regardless of guest depth. This is a reference-oracle limit,
/// not the production recursion ceiling (the JIT uses the guest's guard-paged data stack,
/// §5).
const MAX_CALL_DEPTH: u32 = 256;

/// Run `func` with `args`, consuming up to `*fuel` execution steps.
///
/// Returns the function's result values, or a `Trap`. Decrements `*fuel` per
/// instruction and per branch so that even an infinite loop terminates — important
/// for fuzzing and for never hanging a test.
pub fn run(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    // No capabilities granted: an empty powerbox (any `cap.call` is inert → `CapFault`).
    let mut host = Host::new();
    run_with_host(m, func, args, fuel, &mut host)
}

/// Like [`run`], but with a caller-provided [`Host`] (the powerbox): grant the entry
/// function's capabilities into `host`, pass their handle indices in `args`, then read
/// effects (`host.stdout`, etc.) back afterwards. This is how a capability-using guest
/// is driven (§3c/§3e).
pub fn run_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    let f = m.funcs.get(func as usize).ok_or(Trap::Malformed)?;
    // One linear-memory window per run, zero-initialized and lazily paged. The whole module
    // shares it. The window is a large reserved range (§4 default policy) with only `mapped`
    // backed, so an out-of-`mapped` access faults (detect-and-kill) instead of wrapping.
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data); // §3a/D40 data segments (copy + RO-protect)
        mm
    });
    run_func(f, args, fuel, &mut mem, &m.funcs, host, 0)
}

/// Like [`run`], but seed the window with `init_mem` (its low bytes) and return the final
/// window contents (the same number of bytes) alongside the result. This is the
/// **escape-oracle** path (§18): a *verified* module must keep every access in-window, so a
/// run that completes without trapping must leave a window byte-identical to the JIT's. The
/// non-zero seed makes a divergent (e.g. under-masked) *read* observable, not just a write.
/// With no declared memory the snapshot is empty.
pub fn run_capture(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed.
    run_capture_reserved(m, func, args, fuel, init_mem, DEFAULT_RESERVED_LOG2)
}

/// Like [`run_capture`], but with a host **reservation policy**: confinement masks into
/// `[0, 2^reserved_log2)` while only the declared `1 << size_log2` bytes are backed, so an
/// access into the reserved-but-unmapped tail faults (`Trap::MemoryFault`) instead of wrapping
/// (the deliberate I1 change for the §4 "guard-when-bounded" model). `reserved_log2` is raised
/// to at least `size_log2` (so `0` ⇒ fully mapped). This is the interpreter side of the
/// escape-oracle under the decoupled model and must be driven with the **same** `reserved_log2`
/// as the JIT's [`svm_jit::compile_and_run_capture_reserved`] to stay in differential lockstep.
pub fn run_capture_reserved(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    let f = match m.funcs.get(func as usize) {
        Some(f) => f,
        None => return (Err(Trap::Malformed), Vec::new()),
    };
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data); // §3a/D40 data segments (after the escape-oracle seed)
        mm
    });
    let r = run_func(f, args, fuel, &mut mem, &m.funcs, &mut host, 0);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    (r, snap)
}

/// Like [`run_capture_reserved`], but with a caller-provided [`Host`] (the powerbox), so a
/// `cap.call` to a *granted* handle takes its **success** path while the final-window snapshot
/// still feeds the escape-oracle (§18). Pairs with the JIT's
/// [`svm_jit::compile_and_run_capture_reserved_with_host`]: running both lets the §3e Memory
/// capability's `map`/`unmap`/`protect` effects be byte-compared across backends, not just their
/// return values — a real generative escape-oracle for the capability path.
/// Escape-oracle snapshot span (the `_with_host` capture): byte-compare the low `SNAP_CAP` bytes of
/// the window — *including* reserved-tail pages the guest grew via the Memory cap, not just the
/// backed prefix. **Must match `svm_jit`'s `SNAP_CAP`** so both backends snapshot the same span.
const SNAP_CAP: usize = 1 << 18; // 256 KiB

pub fn run_capture_reserved_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let f = match m.funcs.get(func as usize) {
        Some(f) => f,
        None => return (Err(Trap::Malformed), Vec::new()),
    };
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm
    });
    let r = run_func(f, args, fuel, &mut mem, &m.funcs, host, 0);
    // Snapshot past the backed prefix to also cover reserved-tail pages the guest grew (the §1a
    // growth path), matching the JIT's `_with_host` capture span so the escape-oracle byte-compares
    // them too.
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_window(SNAP_CAP))
        .unwrap_or_default();
    (r, snap)
}

/// One activation record on the **explicit** guest call stack (§12). Reifying the call
/// stack — rather than recursing on the host stack — is the prerequisite for fibers: a
/// fiber's continuation is exactly its `Vec<Frame>`, which a future `suspend` can pause
/// and a `resume` restart. Today the driver still runs each call to completion; nothing
/// observable changes versus the old recursive interpreter.
struct Frame<'a> {
    /// The function this activation is executing.
    f: &'a Func,
    /// Index of the block currently executing.
    block: usize,
    /// Index of the **next** instruction to execute within that block. Saved across a
    /// nested call so the caller resumes just past the `call` when the callee returns.
    inst: usize,
    /// Block-local SSA values produced so far (entry = the call arguments).
    vals: Vec<Value>,
}

fn run_func<'a>(
    f: &'a Func,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    funcs: &'a [Func],
    host: &mut Host,
    depth: u32,
) -> Result<Vec<Value>, Trap> {
    // The reified guest call stack. The entry block's parameters are the function
    // arguments (verifier-checked; copied defensively). `depth` is the base depth of the
    // bottom frame (0 at top level) so the bound covers any future resume above a floor.
    let mut stack: Vec<Frame<'a>> = vec![Frame {
        f,
        block: 0,
        inst: 0,
        vals: args.to_vec(),
    }];

    // Drive the top frame. A `call` pushes a new top and restarts here; a `return` pops
    // and appends results to the caller (which resumes past the call); a tail call
    // replaces the top in place (no growth) — so deep tail recursion stays O(1) frames.
    'frames: loop {
        let top = stack.len() - 1;
        // `block` borrows `funcs` (lifetime `'a`), *not* `stack`, so the loop body is free
        // to push/pop/mutate `stack` while holding it.
        let block = stack[top]
            .f
            .blocks
            .get(stack[top].block)
            .ok_or(Trap::Malformed)?;

        // Execute the remaining instructions of this block.
        while stack[top].inst < block.insts.len() {
            let inst = &block.insts[stack[top].inst];
            step(fuel)?;
            stack[top].inst += 1; // advance first, so a call-return resumes past this inst

            match inst {
                // Non-tail calls push a new frame and switch to it; the callee's results
                // are appended to this frame's `vals` when it returns (see `Return`).
                Inst::Call { func, args } => {
                    let argv = collect(&stack[top].vals, args)?;
                    let callee = funcs.get(*func as usize).ok_or(Trap::Malformed)?;
                    if depth as usize + stack.len() > MAX_CALL_DEPTH as usize {
                        return Err(Trap::StackOverflow);
                    }
                    stack.push(Frame {
                        f: callee,
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                Inst::CallIndirect { ty, idx, args } => {
                    let callee = table_lookup(funcs, as_i32(get(&stack[top].vals, *idx)?)?, ty)?;
                    let argv = collect(&stack[top].vals, args)?;
                    if depth as usize + stack.len() > MAX_CALL_DEPTH as usize {
                        return Err(Trap::StackOverflow);
                    }
                    stack.push(Frame {
                        f: callee,
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                Inst::CapCall {
                    type_id,
                    op,
                    sig,
                    handle,
                    args,
                } => {
                    // Capability call (§3c): resolve the handle in the host-owned table
                    // (mask + type_id/generation check) and dispatch to the mock host.
                    // Args/results cross as i64 slots (the shared host-dispatch ABI).
                    // Synchronous in the reference (the async/submit-complete ABI is §12).
                    let h = as_i32(get(&stack[top].vals, *handle)?)?;
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(val_to_slot(get(&stack[top].vals, *a)?));
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let results = host.cap_dispatch_slots(*type_id, *op, h, &argv, gm)?;
                    for (s, ty) in results.iter().zip(&sig.results) {
                        stack[top].vals.push(slot_to_val(*ty, *s));
                    }
                }
                // Everything else: one value, or none for `Store`/`AtomicStore`.
                other => {
                    if let Some(v) = eval_inst(other, &stack[top].vals, mem)? {
                        stack[top].vals.push(v);
                    }
                }
            }
        }

        step(fuel)?;
        match &block.term {
            Terminator::Br { target, args } => {
                stack[top].vals = collect(&stack[top].vals, args)?;
                stack[top].block = *target as usize;
                stack[top].inst = 0;
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (target, edge_args) = if as_i32(get(&stack[top].vals, *cond)?)? != 0 {
                    (*then_blk, then_args)
                } else {
                    (*else_blk, else_args)
                };
                stack[top].vals = collect(&stack[top].vals, edge_args)?;
                stack[top].block = target as usize;
                stack[top].inst = 0;
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let i = as_i32(get(&stack[top].vals, *idx)?)? as u32 as usize;
                let (target, edge_args) = targets.get(i).unwrap_or(default);
                stack[top].vals = collect(&stack[top].vals, edge_args)?;
                stack[top].block = *target as usize;
                stack[top].inst = 0;
            }
            Terminator::Return(out) => {
                let results = collect(&stack[top].vals, out)?;
                stack.pop();
                match stack.last_mut() {
                    // Caller resumes past the `call` (its `inst` already advanced).
                    Some(caller) => caller.vals.extend(results),
                    None => return Ok(results), // bottom frame returned: the run is done
                }
            }
            Terminator::Unreachable => return Err(Trap::Unreachable),
            // Tail calls replace the top frame in place — no depth growth.
            Terminator::ReturnCall { func, args } => {
                let argv = collect(&stack[top].vals, args)?;
                let callee = funcs.get(*func as usize).ok_or(Trap::Malformed)?;
                stack[top] = Frame {
                    f: callee,
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
            Terminator::ReturnCallIndirect { ty, idx, args } => {
                let callee = table_lookup(funcs, as_i32(get(&stack[top].vals, *idx)?)?, ty)?;
                let argv = collect(&stack[top].vals, args)?;
                stack[top] = Frame {
                    f: callee,
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
        }
    }
}

fn eval_inst(inst: &Inst, vals: &[Value], mem: &mut Option<Mem>) -> Result<Option<Value>, Trap> {
    // `Store` is the only instruction that produces no value.
    if let Inst::Store {
        op,
        addr,
        value,
        offset,
        ..
    } = inst
    {
        let m = mem.as_mut().ok_or(Trap::Malformed)?;
        let a = as_i64(get(vals, *addr)?)? as u64;
        let v = get(vals, *value)?;
        m.store(a, *offset, *op, v)?;
        return Ok(None);
    }
    // §12 atomic store — the other no-result memory op.
    if let Inst::AtomicStore {
        ty,
        addr,
        value,
        offset,
    } = inst
    {
        let m = mem.as_mut().ok_or(Trap::Malformed)?;
        let a = as_i64(get(vals, *addr)?)? as u64;
        let v = get(vals, *value)?;
        m.atomic_store(a, *offset, *ty, v)?;
        return Ok(None);
    }
    let v = match inst {
        Inst::ConstI32(c) => Value::I32(*c),
        Inst::ConstI64(c) => Value::I64(*c),
        Inst::IntBin { ty, op, a, b } => match ty {
            IntTy::I32 => Value::I32(bin32(
                *op,
                as_i32(get(vals, *a)?)?,
                as_i32(get(vals, *b)?)?,
            )?),
            IntTy::I64 => Value::I64(bin64(
                *op,
                as_i64(get(vals, *a)?)?,
                as_i64(get(vals, *b)?)?,
            )?),
        },
        Inst::IntCmp { ty, op, a, b } => {
            let r = match ty {
                IntTy::I32 => cmp32(*op, as_i32(get(vals, *a)?)?, as_i32(get(vals, *b)?)?),
                IntTy::I64 => cmp64(*op, as_i64(get(vals, *a)?)?, as_i64(get(vals, *b)?)?),
            };
            Value::I32(r as i32)
        }
        Inst::IntUn { ty, op, a } => match ty {
            IntTy::I32 => Value::I32(intun32(*op, as_i32(get(vals, *a)?)?)),
            IntTy::I64 => Value::I64(intun64(*op, as_i64(get(vals, *a)?)?)),
        },
        Inst::Eqz { ty, a } => {
            let r = match ty {
                IntTy::I32 => as_i32(get(vals, *a)?)? == 0,
                IntTy::I64 => as_i64(get(vals, *a)?)? == 0,
            };
            Value::I32(r as i32)
        }
        Inst::Convert { op, a } => match op {
            ConvOp::ExtendI32S => Value::I64(as_i32(get(vals, *a)?)? as i64),
            ConvOp::ExtendI32U => Value::I64(as_i32(get(vals, *a)?)? as u32 as i64),
            ConvOp::WrapI64 => Value::I32(as_i64(get(vals, *a)?)? as i32),
        },
        Inst::Select { cond, a, b } => {
            if as_i32(get(vals, *cond)?)? != 0 {
                get(vals, *a)?
            } else {
                get(vals, *b)?
            }
        }
        Inst::ConstF32(bits) => Value::F32(f32::from_bits(*bits)),
        Inst::ConstF64(bits) => Value::F64(f64::from_bits(*bits)),
        Inst::FBin { ty, op, a, b } => match ty {
            FloatTy::F32 => Value::F32(fbin32(
                *op,
                as_f32(get(vals, *a)?)?,
                as_f32(get(vals, *b)?)?,
            )),
            FloatTy::F64 => Value::F64(fbin64(
                *op,
                as_f64(get(vals, *a)?)?,
                as_f64(get(vals, *b)?)?,
            )),
        },
        Inst::FUn { ty, op, a } => match ty {
            FloatTy::F32 => Value::F32(fun32(*op, as_f32(get(vals, *a)?)?)),
            FloatTy::F64 => Value::F64(fun64(*op, as_f64(get(vals, *a)?)?)),
        },
        Inst::FCmp { ty, op, a, b } => {
            let r = match ty {
                FloatTy::F32 => fcmp32(*op, as_f32(get(vals, *a)?)?, as_f32(get(vals, *b)?)?),
                FloatTy::F64 => fcmp64(*op, as_f64(get(vals, *a)?)?, as_f64(get(vals, *b)?)?),
            };
            Value::I32(r as i32)
        }
        Inst::FToISat { op, a } => fto_i(*op, get(vals, *a)?)?,
        Inst::FToITrap { op, a } => trunc_trap(*op, get(vals, *a)?)?,
        Inst::IToFConv { op, a } => i_to_f(*op, get(vals, *a)?)?,
        Inst::PtrAdd { a, b } => {
            Value::I64(as_i64(get(vals, *a)?)?.wrapping_add(as_i64(get(vals, *b)?)?))
        }
        // `ptr.from_int`/`ptr.to_int` are a no-op off-CHERI: pass the i64 through.
        Inst::PtrCast { a, .. } => Value::I64(as_i64(get(vals, *a)?)?),
        Inst::Cast { op, a } => cast(*op, get(vals, *a)?)?,
        // A funcref is just the function index as plain i32 data (§3c).
        Inst::RefFunc { func } => Value::I32(*func as i32),
        Inst::Load {
            op, addr, offset, ..
        } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            m.load(a, *offset, *op)?
        }
        Inst::AtomicLoad { ty, addr, offset } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            m.atomic_load(a, *offset, *ty)?
        }
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            let v = get(vals, *value)?;
            m.atomic_rmw(a, *offset, *ty, *op, v)?
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = as_i64(get(vals, *addr)?)? as u64;
            let exp = get(vals, *expected)?;
            let rep = get(vals, *replacement)?;
            m.atomic_cmpxchg(a, *offset, *ty, exp, rep)?
        }
        // Handled before/around the match; listed for exhaustiveness (no panic).
        Inst::Store { .. }
        | Inst::AtomicStore { .. }
        | Inst::Call { .. }
        | Inst::CallIndirect { .. }
        | Inst::CapCall { .. } => return Ok(None),
    };
    Ok(Some(v))
}

/// Resolve a `call_indirect`: mask the index into the power-of-two-padded function
/// table, then check the selected entry's signature against `ty` (the §3c table
/// type-id check). Masking — not branching — keeps the table load Spectre-v1 safe.
fn table_lookup<'a>(funcs: &'a [Func], idx: i32, ty: &FuncType) -> Result<&'a Func, Trap> {
    let mask = funcs.len().next_power_of_two() - 1;
    let slot = (idx as u32 as usize) & mask;
    match funcs.get(slot) {
        Some(c) if c.params == ty.params && c.results == ty.results => Ok(c),
        _ => Err(Trap::IndirectCallType),
    }
}

fn fbin32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin32(a, b),
        FBinOp::Max => fmax32(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fbin64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin64(a, b),
        FBinOp::Max => fmax64(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fun32(op: FUnOp, a: f32) -> f32 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fun64(op: FUnOp, a: f64) -> f64 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fcmp32(op: FCmpOp, a: f32, b: f32) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

fn fcmp64(op: FCmpOp, a: f64, b: f64) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

// wasm min/max: NaN propagates; for ±0, min prefers -0 and max prefers +0.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

// Float→int casts are saturating with NaN→0 (Rust `as` matches wasm `trunc_sat`).
fn fto_i(op: FToI, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        FToI::F32I32S => Value::I32(as_f32(v)? as i32),
        FToI::F32I32U => Value::I32(as_f32(v)? as u32 as i32),
        FToI::F32I64S => Value::I64(as_f32(v)? as i64),
        FToI::F32I64U => Value::I64(as_f32(v)? as u64 as i64),
        FToI::F64I32S => Value::I32(as_f64(v)? as i32),
        FToI::F64I32U => Value::I32(as_f64(v)? as u32 as i32),
        FToI::F64I64S => Value::I64(as_f64(v)? as i64),
        FToI::F64I64U => Value::I64(as_f64(v)? as u64 as i64),
    })
}

/// Trapping float→int conversion (`trunc`, vs the saturating `trunc_sat`): NaN and
/// out-of-range inputs trap. Work in `f64` (promoting `f32` is exact), and trap
/// unless the truncation toward zero fits the target — `f > MIN-1 && f < MAX+1`
/// (using the exact float boundary constants; the `i64` signed lower bound is
/// closed because `-2^63 - 1` is not representable and rounds to `-2^63`).
fn trunc_trap(op: FToI, v: Value) -> Result<Value, Trap> {
    let (from, to, signed) = op.parts();
    let f: f64 = match from {
        FloatTy::F32 => as_f32(v)? as f64,
        FloatTy::F64 => as_f64(v)?,
    };
    if f.is_nan() {
        return Err(Trap::BadConversion);
    }
    // Bounds are written as explicit comparisons so the open-vs-closed distinction is
    // visible: the i64-signed *lower* bound is closed (`>=`) because `-2^63 - 1` is
    // not representable and rounds to `-2^63`; the rest are open.
    #[allow(clippy::manual_range_contains)]
    let in_range = match (to, signed) {
        (IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (IntTy::I64, true) => f >= -9_223_372_036_854_775_808.0 && f < 9_223_372_036_854_775_808.0,
        (IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return Err(Trap::BadConversion);
    }
    // In range, so the cast is exact (truncating toward zero, no saturation).
    Ok(match (to, signed) {
        (IntTy::I32, true) => Value::I32(f as i32),
        (IntTy::I32, false) => Value::I32(f as u32 as i32),
        (IntTy::I64, true) => Value::I64(f as i64),
        (IntTy::I64, false) => Value::I64(f as u64 as i64),
    })
}

fn i_to_f(op: IToF, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        IToF::I32F32S => Value::F32(as_i32(v)? as f32),
        IToF::I32F32U => Value::F32(as_i32(v)? as u32 as f32),
        IToF::I64F32S => Value::F32(as_i64(v)? as f32),
        IToF::I64F32U => Value::F32(as_i64(v)? as u64 as f32),
        IToF::I32F64S => Value::F64(as_i32(v)? as f64),
        IToF::I32F64U => Value::F64(as_i32(v)? as u32 as f64),
        IToF::I64F64S => Value::F64(as_i64(v)? as f64),
        IToF::I64F64U => Value::F64(as_i64(v)? as u64 as f64),
    })
}

fn cast(op: CastOp, v: Value) -> Result<Value, Trap> {
    Ok(match op {
        CastOp::Demote => Value::F32(as_f64(v)? as f32),
        CastOp::Promote => Value::F64(as_f32(v)? as f64),
        CastOp::ReinterpI32F32 => Value::F32(f32::from_bits(as_i32(v)? as u32)),
        CastOp::ReinterpF32I32 => Value::I32(as_f32(v)?.to_bits() as i32),
        CastOp::ReinterpI64F64 => Value::F64(f64::from_bits(as_i64(v)? as u64)),
        CastOp::ReinterpF64I64 => Value::I64(as_f64(v)?.to_bits() as i64),
    })
}

// ----------------------------------------------------------------------------
// Capabilities — the host-owned handle table + a deterministic mock powerbox
// (§3c index model, §3e MVP interface set). This is the reference oracle's
// stand-in for real host capabilities: deterministic, in-process, so it can be a
// differential oracle. The *security* of the model lives in `Host::resolve`
// (use-site mask + type_id + generation check → forged indices are inert).
// ----------------------------------------------------------------------------

/// MVP interface type-ids (§3e). Phase-1: a `type_id` is just a small constant a
/// handle-table entry carries and `cap.call` re-checks. (A module-level interface
/// section that globalizes ids across linked modules is deferred to §13.)
pub mod iface {
    /// `Stream` — byte stream: op 0 `read`, op 1 `write`, op 2 `close` (§3e D43).
    pub const STREAM: u32 = 0;
    /// `Exit` — lifecycle: op 0 `exit(code)` (noreturn).
    pub const EXIT: u32 = 1;
    /// `Clock` — op 0 `now(clock_id) -> i64` nanoseconds.
    pub const CLOCK: u32 = 2;
    /// `Memory` — op 0 `map`, 1 `unmap`, 2 `protect`, 3 `page_size` (§3e; real page protection —
    /// see `Mem`).
    pub const MEMORY: u32 = 3;
    /// `SharedRegion` — a host-backed memory object aliased into the window (§13). op 0
    /// `map(window_offset, region_offset, len, prot)` aliases the region's pages into the window
    /// (the same backing may be mapped at *multiple* window offsets → zero-overhead aliasing, the
    /// magic-ring-buffer primitive); op 1 `unmap(window_offset, len)` drops the alias; op 2
    /// `len() -> i64` reports the region size; op 3 `page_size() -> i64`. Granting the handle is how
    /// two domains come to share memory; `create`/`grant` (guest-minted regions, cross-domain) are a
    /// §14 follow-up — today regions are host-granted, like `Memory`.
    pub const SHARED_REGION: u32 = 4;
}

/// Negative-errno values returned by capability ops (§3e D42): `< 0` is `-errno`,
/// `>= 0` is success. Errors do **not** trap — traps stay reserved for escape/fatal.
const EFAULT: i64 = -14; // buffer not fully within the window
const EINVAL: i64 = -22; // bad op / argument

/// Cap ABI `prot` bits for the `Memory` capability (§3e): the low two bits of the `i32`
/// argument. There is no `EXEC` bit — guest data is never executed as code (§3c).
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;

/// A §13 `SharedRegion`'s backing — a host-owned shared object aliased into a window at one or more
/// offsets. The reference (interpreter) backing is a plain Rust buffer ([`VecBacking`]); a flat-window
/// backend (the JIT) supplies one wrapping a real OS shared-memory object (memfd / file mapping) whose
/// [`SharedBacking::os_fd`] it `mmap`s for true hardware aliasing. Cloning the `Rc` shares the *same*
/// object, so two mappings of it alias.
pub trait SharedBacking {
    /// Region size in bytes.
    fn size(&self) -> u64;
    /// Read one region-relative byte (out of range ⇒ 0).
    fn read_byte(&self, off: u64) -> u8;
    /// Write one region-relative byte (out of range ⇒ ignored). Interior-mutable: a region is shared
    /// (`Rc`), so writes go through `&self`.
    fn write_byte(&self, off: u64, b: u8);
    /// An OS shared-memory handle a flat-window backend can `mmap` for real aliasing; `None` for the
    /// pure-Rust reference backing (the interpreter models aliasing in software instead). Unix
    /// (`memfd`/`shm`); the Windows analogue is [`os_section`](SharedBacking::os_section).
    fn os_fd(&self) -> Option<i32> {
        None
    }

    /// A Windows section handle (from `CreateFileMapping`) a flat-window backend maps into the window
    /// via `MapViewOfFile3` for real aliasing — the Windows analogue of [`os_fd`](SharedBacking::os_fd).
    /// Carried as an `isize` (a `HANDLE` is pointer-sized) to keep this trait platform-clean; `None`
    /// for the pure-Rust reference backing. Only the Windows JIT path consumes it.
    fn os_section(&self) -> Option<isize> {
        None
    }
}

/// A reference to a shared region backing (see [`SharedBacking`]); cloning shares the same object.
pub type RegionBacking = Rc<dyn SharedBacking>;

/// The reference [`SharedBacking`]: a plain in-process buffer. The interpreter models aliasing by
/// reading/writing this shared buffer through several `Backed` pages.
struct VecBacking(RefCell<Vec<u8>>);

impl SharedBacking for VecBacking {
    fn size(&self) -> u64 {
        self.0.borrow().len() as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        self.0.borrow().get(off as usize).copied().unwrap_or(0)
    }
    fn write_byte(&self, off: u64, b: u8) {
        if let Some(s) = self.0.borrow_mut().get_mut(off as usize) {
            *s = b;
        }
    }
}

/// The guest window a capability handler borrows `(ptr, len)` buffers from (§7). Both
/// the interpreter's lazily-paged [`Mem`] and a JIT's flat window implement this, so a
/// single host dispatch ([`Host::cap_dispatch`]) serves both backends. The methods
/// bounds-check `[ptr, ptr+len) ⊆ [0, size)` and return `None` (→ `-EFAULT`) otherwise.
pub trait GuestMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>>;
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()>;

    /// `Memory` capability ops (§3e): (re)commit / decommit / re-protect window pages. `offset`
    /// is page-aligned and `[offset, offset+len)` window-relative; `prot` is `READ|WRITE`. Each
    /// returns `0` or a negative errno (`-EINVAL`). The default is a success no-op — overridden
    /// by the interpreter's paged [`Mem`] (the reference semantics); a flat-window backend
    /// (e.g. a JIT) wires its own `mprotect`-backed implementation.
    fn map(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }
    fn unmap(&mut self, _offset: u64, _len: u64) -> i64 {
        0
    }
    fn protect(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }

    /// `SharedRegion` op 0 `map` (§13): alias `backing`'s `[region_off, region_off+len)` pages into
    /// the window at `[win_off, win_off+len)` with `prot`. The same `region`/`backing` mapped at two
    /// window offsets makes both ranges name the *same* bytes (zero-overhead aliasing). `0` or a
    /// negative errno. The default rejects it (`-EINVAL`): only the reference paged [`Mem`] models
    /// aliasing today; a flat-window backend wires its own shared mapping (§13 slice 2).
    fn map_region(
        &mut self,
        _win_off: u64,
        _region_off: u64,
        _len: u64,
        _prot: i32,
        _region: u32,
        _backing: RegionBacking,
    ) -> i64 {
        EINVAL
    }

    /// `Memory` op 3 `page_size() -> i64`: the host MMU page granularity this window is managed in —
    /// the unit `map`/`unmap`/`protect` round to. A guest queries it to align its own allocator to
    /// the real page (4 KiB / 16 KiB / …) and adapt, instead of assuming a fixed size. The default
    /// reports the host page; the paged [`Mem`] and the JIT's `MprotectWindow` override it with the
    /// exact value they round to, so the two backends stay in differential lockstep.
    fn page_size(&self) -> i64 {
        host_page_size() as i64
    }

    /// `SharedRegion` op 3 `page_size() -> i64`: the granularity a `SharedRegion` map aligns to —
    /// the host page on unix, the **allocation granularity** (64 KiB) on Windows, which
    /// `MapViewOfFile3` requires. Distinct from [`page_size`](GuestMem::page_size) (the protection
    /// granularity) so a guest aligns its region maps to a value that works on every backend. The
    /// default ([`host_region_granularity`]) is correct for both the paged [`Mem`] and the JIT's
    /// flat window, so the two stay in §13 lockstep without an override.
    fn region_page_size(&self) -> i64 {
        host_region_granularity() as i64
    }
}

/// A [`GuestMem`] over a flat, contiguous window slice — the JIT's representation. The
/// slice may include trailing guard bytes; `size` is the *logical* window so the §7
/// bounds check matches the interpreter exactly.
pub struct WindowMem<'a> {
    window: &'a mut [u8],
    size: u64,
}

impl<'a> WindowMem<'a> {
    pub fn new(window: &'a mut [u8], size: u64) -> WindowMem<'a> {
        WindowMem { window, size }
    }
}

impl GuestMem for WindowMem<'_> {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        if end > self.size {
            return None;
        }
        Some(self.window[ptr as usize..end as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let end = ptr.checked_add(data.len() as u64)?;
        if end > self.size {
            return None;
        }
        self.window[ptr as usize..end as usize].copy_from_slice(data);
        Some(())
    }
}

/// Which standard stream a `Stream` handle is bound to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamRole {
    In,
    Out,
    Err,
}

/// The host-side object a handle-table entry dispatches to — the mock equivalent of
/// §3c's `(methods, object)`. The guest never names or writes this (it lives in host
/// memory); it is selected only by a *granted* handle index.
#[derive(Clone, Copy, Debug)]
enum Binding {
    Stream(StreamRole),
    Exit,
    Clock,
    Memory,
    /// A §13 `SharedRegion` handle, carrying the index of its backing in [`Host::regions`]. The
    /// backing (not the index) is the shared object; mapping it at several window offsets aliases.
    SharedRegion(u32),
}

/// One handle-table slot (§3c): host-owned, guest-unwritable. `generation` is
/// per-slot and only advances on (re)grant, so a closed handle's value can never
/// alias a later grant of the same slot (ABA-safe use-after-close detection, D37).
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    generation: u32,
    entry: Option<Binding>,
    type_id: u32,
}

/// `log2` of the handle-table capacity. A handle value packs `(generation, slot)`:
/// `slot = h & (cap-1)`, `generation = h >> CAP_LOG2`.
const CAP_LOG2: u32 = 8;
const CAP: usize = 1 << CAP_LOG2;

/// The host: the **host-owned handle table** (the powerbox) plus deterministic mock
/// capability state (captured stdio, a monotonic clock). Construct with [`Host::new`],
/// `grant_*` the initial capabilities, then pass to [`run_with_host`]; afterwards read
/// back `stdout`/`stderr`. Deterministic by design so it serves as a §18 oracle.
pub struct Host {
    table: Vec<Slot>, // CAP slots, host-owned
    /// Bytes a `Stream{In}` handle's `read` draws from.
    pub stdin: Vec<u8>,
    stdin_pos: usize,
    /// Bytes written by `Stream{Out}` / `Stream{Err}` `write`s.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// Monotonic nanosecond counter; each `Clock.now` returns it then advances by one,
    /// so reads are deterministic and strictly increasing.
    pub clock_ns: i64,
    /// §13 `SharedRegion` backings, indexed by the id a [`Binding::SharedRegion`] carries. Each is a
    /// shared host buffer; aliasing a region into several window offsets clones this `Rc`.
    regions: Vec<RegionBacking>,
}

impl Default for Host {
    fn default() -> Host {
        Host::new()
    }
}

impl Host {
    pub fn new() -> Host {
        Host {
            table: vec![Slot::default(); CAP],
            stdin: Vec::new(),
            stdin_pos: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            clock_ns: 0,
            regions: Vec::new(),
        }
    }

    /// Install a host binding in a free slot and return the guest handle — a forgeable
    /// `i32` index encoding `(generation, slot)`. This is how the powerbox (and, later,
    /// attenuation) hands authority to the guest (§3c). Panics only if the table is
    /// full (a host bug, not reachable from guest code).
    fn grant(&mut self, type_id: u32, binding: Binding) -> i32 {
        let slot = self
            .table
            .iter()
            .position(|s| s.entry.is_none())
            .expect("handle table full");
        let s = &mut self.table[slot];
        s.generation = s.generation.wrapping_add(1); // advance per (re)grant (ABA-safe)
        s.entry = Some(binding);
        s.type_id = type_id;
        ((s.generation << CAP_LOG2) | slot as u32) as i32
    }

    /// Grant a `Stream` capability bound to `role` (a powerbox stdio grant, §3e).
    pub fn grant_stream(&mut self, role: StreamRole) -> i32 {
        self.grant(iface::STREAM, Binding::Stream(role))
    }
    pub fn grant_exit(&mut self) -> i32 {
        self.grant(iface::EXIT, Binding::Exit)
    }
    pub fn grant_clock(&mut self) -> i32 {
        self.grant(iface::CLOCK, Binding::Clock)
    }
    pub fn grant_memory(&mut self) -> i32 {
        self.grant(iface::MEMORY, Binding::Memory)
    }

    /// Grant a §13 `SharedRegion` capability backed by a fresh `len`-byte zero-filled host buffer,
    /// returning its handle. The guest `map`s it into its window (op 0) — at one or more offsets — to
    /// access the shared bytes as ordinary masked loads/stores. (Guest-minted regions and
    /// cross-domain `grant` are a §14 follow-up; this models the host↔guest data plane.)
    pub fn grant_shared_region(&mut self, len: usize) -> i32 {
        self.grant_shared_region_backed(Rc::new(VecBacking(RefCell::new(vec![0u8; len]))))
    }

    /// Grant a §13 `SharedRegion` over a caller-supplied [`SharedBacking`] — how a flat-window
    /// backend installs a region whose `os_fd` it can `mmap` for real hardware aliasing (the JIT
    /// side of the §13 differential). The pure-Rust [`grant_shared_region`] is the common case.
    pub fn grant_shared_region_backed(&mut self, backing: RegionBacking) -> i32 {
        let id = self.regions.len() as u32;
        self.regions.push(backing);
        self.grant(iface::SHARED_REGION, Binding::SharedRegion(id))
    }

    /// Close a handle (§3c): free the slot but keep its generation, so the old handle
    /// value is now a dead generation and any later `cap.call` on it traps (D37).
    pub fn close(&mut self, handle: i32) {
        let slot = (handle as u32 as usize) & (CAP - 1);
        self.table[slot].entry = None;
    }

    /// Resolve a handle at a `cap.call` use site (§3c) — **the security hinge**: mask
    /// the index into the host-owned table (never branch), then re-check the entry's
    /// interface `type_id` and `generation`. A forged / closed / wrong-type index is
    /// inert: it faults, or at worst selects one of *this domain's own* granted
    /// `type_id` capabilities. The guest never supplies the binding.
    fn resolve(&self, handle: i32, type_id: u32) -> Result<Binding, Trap> {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1); // mask, not branch (Spectre-v1 safe)
        let gen = h >> CAP_LOG2;
        let s = &self.table[slot];
        match s.entry {
            Some(b) if s.type_id == type_id && s.generation == gen => Ok(b),
            _ => Err(Trap::CapFault),
        }
    }

    /// Dispatch a `cap.call` (§3c): resolve the handle, then run the mock operation.
    /// Returns the op's result values (negative-errno encoded in an `i64` for the
    /// fallible ops, §3e D42), or a `Trap` for escape/exit. `mem` backs buffer args.
    /// Dispatch a `cap.call` (§3c): resolve the handle in the host-owned table, then run
    /// the bound capability op. Public and **slot-based** (`i64` per scalar; `i32` in
    /// the low bits) so both backends drive the same handlers without per-arg type tags
    /// — the interpreter converts its `Value`s, a JIT passes its slots directly. `mem`
    /// is `None` when the module declares no memory (buffer ops then return `-EFAULT`).
    pub fn cap_dispatch_slots(
        &mut self,
        type_id: u32,
        op: u32,
        handle: i32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        match self.resolve(handle, type_id)? {
            Binding::Stream(role) => self.stream_op(role, op, args, mem),
            Binding::Exit => {
                // op 0: exit(code: i32) — noreturn. Propagate as a (non-error) trap.
                let code = *args.first().ok_or(Trap::Malformed)? as i32;
                Err(Trap::Exit(code))
            }
            Binding::Clock => {
                // op 0: now(clock_id) -> i64 nanoseconds (deterministic, increasing).
                let now = self.clock_ns;
                self.clock_ns = self.clock_ns.wrapping_add(1);
                Ok(vec![now])
            }
            Binding::Memory => {
                // map(off,len,prot) / unmap(off,len) / protect(off,len,prot) on the window's
                // pages (§3e). With no window there is nothing to address (-EINVAL); the effect
                // is applied to whichever backend's memory `mem` wraps (interp `Mem` here, a
                // JIT's flat window via its own impl), keeping the two in differential lockstep.
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                let off = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let prot = *args.get(2).unwrap_or(&0) as i32;
                Ok(vec![match op {
                    0 => mem.map(off, len, prot),
                    1 => mem.unmap(off, len),
                    2 => mem.protect(off, len, prot),
                    3 => mem.page_size(),
                    _ => EINVAL,
                }])
            }
            Binding::SharedRegion(region) => {
                // §13: alias the host-backed region into the window. `map` (op 0) at several offsets
                // aliases the same bytes; loads/stores then go through the ordinary masked path.
                let Some(backing) = self.regions.get(region as usize).cloned() else {
                    return Ok(vec![EINVAL]);
                };
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                Ok(vec![match op {
                    0 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let region_off = *args.get(1).unwrap_or(&0) as u64;
                        let len = *args.get(2).unwrap_or(&0) as u64;
                        let prot = *args.get(3).unwrap_or(&0) as i32;
                        mem.map_region(win_off, region_off, len, prot, region, backing)
                    }
                    1 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let len = *args.get(1).unwrap_or(&0) as u64;
                        mem.unmap(win_off, len)
                    }
                    2 => backing.size() as i64,
                    3 => mem.region_page_size(),
                    _ => EINVAL,
                }])
            }
        }
    }

    /// `Stream` ops (§3e D43): 0 `read`, 1 `write`, 2 `close`. Buffers are `(ptr,len)`,
    /// borrow-only — the host reads/writes the guest window in place after the §7
    /// trampoline bounds-checks `[ptr,ptr+len) ⊆ [0,size)` (violation → `-EFAULT`).
    fn stream_op(
        &mut self,
        role: StreamRole,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let ret = |v: i64| Ok(vec![v]);
        match op {
            0 => {
                // read(buf, len) -> bytes read (>=0) or -errno; only stdin is readable.
                if role != StreamRole::In {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
                let n = (len as usize).min(avail.len());
                let chunk = avail[..n].to_vec();
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                if m.write_bytes(ptr, &chunk).is_none() {
                    return ret(EFAULT);
                }
                self.stdin_pos += n;
                ret(n as i64)
            }
            1 => {
                // write(buf, len) -> bytes written (>=0) or -errno; stdin is not writable.
                let sink = match role {
                    StreamRole::Out => &mut self.stdout,
                    StreamRole::Err => &mut self.stderr,
                    StreamRole::In => return ret(EINVAL),
                };
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                match m.read_bytes(ptr, len) {
                    Some(bytes) => {
                        sink.extend_from_slice(&bytes);
                        ret(len as i64)
                    }
                    None => ret(EFAULT),
                }
            }
            2 => ret(0), // close: no-op in the MVP (exit reclaims all)
            _ => ret(EINVAL),
        }
    }
}

// ----------------------------------------------------------------------------
// Linear memory — the confinement-masking *reference* (§4, invariant I1)
// ----------------------------------------------------------------------------

/// The **host** page size — the granularity of the protection model (RO/unmap) *and* the lazy
/// backing-store chunk. Queried so the interpreter's protection granularity matches the JIT's real
/// `mprotect` on the same host (§4 "pin page size", host-page default); both backends query the
/// same value, so they agree page-for-page on any platform (4 KiB / 16 KiB / …). Lazy paging keeps
/// interpreter memory bounded by what a (fuel-limited) run touches, so a huge declared window never
/// eagerly allocates — safe to fuzz.
fn host_page_size() -> u64 {
    match page_size::get() as u64 {
        0 => 4096,
        p => p,
    }
}

/// The granularity a `SharedRegion` map (§13) aligns to — distinct from [`host_page_size`] (the
/// protection granularity) because a *shared mapping* is coarser on Windows. On unix this is the
/// host page (`mmap(MAP_FIXED)` aliases at page granularity); on Windows it is the **allocation
/// granularity** (64 KiB), which `MapViewOfFile3` *requires* for both the placement address and the
/// section offset. Both the interpreter reference and the JIT's flat window report this for
/// `SharedRegion` op 3 (`region_page_size`), so a guest aligns its region maps to a single value that
/// works on every backend and the §13 differential stays in lockstep. `page_size::get_granularity`
/// returns `dwAllocationGranularity` on Windows and the page size on unix.
pub fn host_region_granularity() -> u64 {
    match page_size::get_granularity() as u64 {
        0 => host_page_size(),
        g => g,
    }
}

/// Explicit per-page state in the guest-visible address space (§3e Memory cap / §4).
///
/// A page absent from the map takes the **default for its region**: read+write inside the
/// initial backed prefix `[0, mapped)`, and *unmapped* in the reserved tail `[mapped, reserved)`
/// — so growth into the tail must be made explicit by a `map` (a [`PageProt::Rw`] entry). This is
/// what lets the guest `map`/`unmap`/`protect` sparsely across the whole reserved window (the §1a
/// "sparse address space / lazy page supply" capability), in lockstep with the JIT's real page
/// tables (an uncommitted page is `PROT_NONE` there and faults identically).
///
/// A committed *anonymous* page is zero-filled and lives in [`Mem::pages`]; a [`PageProt::Backed`]
/// page's bytes instead live in a §13 `SharedRegion` buffer (keyed in [`Mem::regions`]) — the
/// primitive behind aliasing / the magic-ring-buffer trick. Crucially the access path
/// ([`Mem::byte`]/[`Mem::set_byte`]) just redirects where a page's bytes live; loads/stores stay
/// ordinary masked accesses (zero overhead), exactly as §13 specifies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageProt {
    /// Explicitly `map`ped read-write — committed even in the reserved tail (where *absent* would
    /// mean unmapped). Within the initial prefix, plain read-write is left *absent* (the default),
    /// so this entry only appears for grown/re-committed pages.
    Rw,
    /// `protect`ed read-only: reads succeed, a store faults (the D40 const-segment mechanism).
    Ro,
    /// `unmap`ped: any access faults.
    Unmapped,
    /// §13 aliased page: its bytes live at `region_off` in the `SharedRegion` `region`
    /// ([`Mem::regions`]), not in an anonymous [`Mem::pages`] entry. `writable` mirrors the map
    /// `prot` (a store to a read-only alias faults). Two pages with the same `region` (mapped at
    /// different window offsets) name the same backing → aliasing.
    Backed {
        region: u32,
        region_off: u64,
        writable: bool,
    },
}

/// A guest linear-memory window. Confinement itself lives in [`svm_mask::Window`]
/// (the isolated, separately-fuzzed security unit, §4); `Mem` owns the lazily paged backing
/// store, threads accesses through that confinement, and carries the guest-visible page
/// protection map (`map`/`unmap`/`protect`, §3e). This is the semantics the JIT is
/// differential-tested against (§18).
struct Mem {
    window: Window,
    /// Host page size (`host_page_size()`): protection + storage-chunk granularity. Cached per
    /// `Mem` so every method shares the one host-queried value (matches the JIT's `mprotect`).
    page: u64,
    pages: BTreeMap<u64, Vec<u8>>,
    /// Page index (`offset / page`) ⇒ explicit page state. A page absent from the map takes its
    /// region default: read+write inside the initial prefix `[0, mapped)`, unmapped in the
    /// reserved tail `[mapped, reserved)`. Entries appear for `protect`ed (`Ro`), `unmap`ped
    /// (`Unmapped`), and grown/re-committed tail (`Rw`) pages — anywhere in `[0, reserved)`.
    prot: BTreeMap<u64, PageProt>,
    /// §13 `SharedRegion` backings this window has aliased in, keyed by region id (the bytes a
    /// [`PageProt::Backed`] page redirects to). A clone of the `Host`'s `Rc`, so two windows — or
    /// two offsets in one window — that map the same region share the *same* bytes.
    regions: BTreeMap<u32, RegionBacking>,
}

impl Mem {
    /// A window whose mask domain is `1 << reserved_log2` bytes but whose backed region is the
    /// declared `1 << mapped_log2` prefix; an access into the reserved-but-unmapped tail faults
    /// (the §4 "guard-when-bounded" model). `reserved_log2` is raised to at least `mapped_log2`,
    /// so passing `0` yields a fully-mapped window. Lazy paging means a huge mask domain (or
    /// reservation) never eagerly allocates.
    fn with_reservation(reserved_log2: u8, mapped_log2: u8) -> Mem {
        let reserved_log2 = reserved_log2.max(mapped_log2);
        Mem {
            window: Window::with_mapped(reserved_log2, 1u64 << mapped_log2.min(63)),
            page: host_page_size(),
            pages: BTreeMap::new(),
            prot: BTreeMap::new(),
            regions: BTreeMap::new(),
        }
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed. A page
    /// absent from the map takes its region default — read+write in the initial prefix
    /// `[0, mapped)`, unmapped in the reserved tail (growth must be an explicit `map`).
    fn page_access(&self, page: u64) -> Option<bool> {
        match self.prot.get(&page) {
            Some(PageProt::Rw) => Some(true),
            Some(PageProt::Ro) => Some(false),
            Some(PageProt::Backed { writable, .. }) => Some(*writable),
            Some(PageProt::Unmapped) => None,
            None => (page * self.page < self.window.mapped()).then_some(true),
        }
    }

    /// Enforce the page state for a `width`-byte access at confined offset `base`: any access to an
    /// unmapped page, or a store to a read-only page, faults (§4/§5). Fast-pathed when the access
    /// lies wholly in the committed prefix and no page has been re-protected (the common case), so
    /// unprotected windows pay nothing.
    fn check_prot(&self, base: u64, width: u32, write: bool) -> Result<(), Trap> {
        let last = base + width as u64 - 1;
        if self.prot.is_empty() && last < self.window.mapped() {
            return Ok(());
        }
        for page in (base / self.page)..=(last / self.page) {
            match self.page_access(page) {
                None => return Err(Trap::MemoryFault), // unmapped
                Some(false) if write => return Err(Trap::MemoryFault), // read-only store
                _ => {}
            }
        }
        Ok(())
    }

    /// Confine the final effective address into `[0, reserved)` (the masking security op, §4) and
    /// reject a `width`-byte access that would overrun the reserved domain. Per-page committed-ness
    /// is enforced separately by [`Mem::check_prot`] (the functional bound), so a masked-but-
    /// uncommitted page faults there — matching the JIT's `PROT_NONE` page tables.
    fn confine_checked(&self, addr: u64, offset: u64, width: u32) -> Result<u64, Trap> {
        let base = self.window.confine(addr, offset);
        match base.checked_add(width as u64) {
            Some(end) if end <= self.window.reserved() => Ok(base),
            _ => Err(Trap::MemoryFault),
        }
    }

    fn load(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Value, Trap> {
        let (_, rty, width, signed) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, false)?;
        let raw = self.read_le(base, width);
        Ok(decode_loaded(rty, width, signed, raw))
    }

    fn store(&mut self, addr: u64, offset: u64, op: StoreOp, v: Value) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, true)?;
        // `write_le` keeps only the low `width` bytes, so narrow stores truncate.
        self.write_le(base, width, store_bits(v));
        Ok(())
    }

    /// §12 atomics share the confinement + page-protection path with `load`/`store`, and add a
    /// **natural-alignment** requirement: a misaligned effective address traps (`MemoryFault`). The
    /// window base and mask domain are width-aligned, so checking the confined address suffices.
    /// Single-threaded, an atomic's *value* semantics equal the non-atomic op; the JIT lowers these
    /// to hardware atomics so they stay correct once threads exist (§12). All operate on the full
    /// `ty` width (`i32`/`i64`).
    fn check_align(&self, base: u64, width: u32) -> Result<(), Trap> {
        if base.is_multiple_of(width as u64) {
            Ok(())
        } else {
            Err(Trap::MemoryFault)
        }
    }

    fn atomic_load(&self, addr: u64, offset: u64, ty: IntTy) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, false)?;
        Ok(atomic_decode(ty, self.read_le(base, width)))
    }

    fn atomic_store(&mut self, addr: u64, offset: u64, ty: IntTy, v: Value) -> Result<(), Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        self.write_le(base, width, store_bits(v));
        Ok(())
    }

    /// Read the old value, apply `op` with `v`, write the result back, return the **old** value.
    fn atomic_rmw(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        op: AtomicRmwOp,
        v: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let old = self.read_le(base, width);
        let new = atomic_rmw_apply(ty, op, old, store_bits(v));
        self.write_le(base, width, new);
        Ok(atomic_decode(ty, old))
    }

    /// If `*addr == expected`, write `replacement`; always return the **old** value.
    fn atomic_cmpxchg(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        expected: Value,
        replacement: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let old = self.read_le(base, width); // already the low `width` bytes, zero-extended
        if old == (store_bits(expected) & width_mask(width)) {
            self.write_le(base, width, store_bits(replacement));
        }
        Ok(atomic_decode(ty, old))
    }

    /// Validate a `map`/`unmap`/`protect` range (§3e): the offset must be page-aligned and the
    /// whole `[offset, offset+len)` must lie within the **reserved** window `[0, reserved)` — the
    /// guest may now grow into the reserved tail `[mapped, reserved)`, not just the initial backed
    /// prefix. Returns the inclusive page-index range it covers, or `Err(EINVAL)`.
    fn prot_pages(&self, offset: u64, len: u64) -> Result<core::ops::RangeInclusive<u64>, i64> {
        if len == 0 || !offset.is_multiple_of(self.page) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.window.reserved() {
            return Err(EINVAL);
        }
        Ok((offset / self.page)..=((end - 1) / self.page)) // len need not be a page multiple; round up
    }

    /// Set one page's protection from cap `prot` bits: `WRITE` ⇒ read+write, `READ` only ⇒
    /// read-only, neither ⇒ unmapped. A read-write page in the initial prefix is left *absent*
    /// (its default); in the reserved tail it needs an explicit [`PageProt::Rw`] entry, since
    /// *absent* there means unmapped.
    fn set_prot(&mut self, page: u64, prot: i32) {
        if prot & PROT_WRITE != 0 {
            if page * self.page < self.window.mapped() {
                self.prot.remove(&page); // read+write is the prefix default (no entry)
            } else {
                self.prot.insert(page, PageProt::Rw); // explicit commit in the reserved tail
            }
        } else if prot & PROT_READ != 0 {
            self.prot.insert(page, PageProt::Ro);
        } else {
            self.prot.insert(page, PageProt::Unmapped);
        }
    }

    /// Place initialized data segments at instantiation (§3a / D40): write every segment's bytes,
    /// then mark the pages of each `readonly` segment read-only (so the init writes themselves
    /// don't fault). RO protection is page-granular, so a producer keeps RO data on its own pages
    /// (the verifier already bounds each segment to `[0, size)`).
    fn init_data(&mut self, data: &[Data]) {
        for d in data {
            for (i, &b) in d.bytes.iter().enumerate() {
                self.set_byte(d.offset + i as u64, b);
            }
        }
        for d in data {
            if d.readonly && !d.bytes.is_empty() {
                let last = d.offset + d.bytes.len() as u64 - 1;
                for page in (d.offset / self.page)..=(last / self.page) {
                    self.prot.insert(page, PageProt::Ro);
                }
            }
        }
    }

    /// Every page touched by `[ptr, ptr+len)` is committed (and writable, when `write`), and the
    /// range stays within `[0, reserved)`. The §7 borrow check: a buffer straddling an unmapped or
    /// (for writes) read-only page is rejected (`-EFAULT`), and grown tail pages are accepted.
    fn range_committed(&self, ptr: u64, len: u64, write: bool) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        if end > self.window.reserved() {
            return false;
        }
        if len == 0 {
            return true;
        }
        (ptr / self.page..=(end - 1) / self.page)
            .all(|page| matches!(self.page_access(page), Some(w) if w || !write))
    }

    /// Borrow-validate and read a `(ptr, len)` capability buffer (§7): every page of
    /// `[ptr, ptr+len)` must be committed. Returns the bytes, or `None` (→ `-EFAULT`).
    /// Confinement holds regardless; this explicit check is the recoverable guest-bug
    /// path, not a safety boundary.
    fn read_bytes_impl(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.range_committed(ptr, len, false) {
            return None;
        }
        Some((0..len).map(|k| self.byte(ptr + k)).collect())
    }

    /// Borrow-validate and write a `(ptr, len)` capability buffer (§7): every page must be
    /// committed and writable. `None` → `-EFAULT`.
    fn write_bytes_impl(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        if !self.range_committed(ptr, data.len() as u64, true) {
            return None;
        }
        for (k, b) in data.iter().enumerate() {
            self.set_byte(ptr + k as u64, *b);
        }
        Some(())
    }

    fn read_le(&self, base: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            raw |= (self.byte(base + k) as u64) << (8 * k);
        }
        raw
    }

    fn write_le(&mut self, base: u64, width: u32, raw: u64) {
        for k in 0..width as u64 {
            self.set_byte(base + k, (raw >> (8 * k)) as u8);
        }
    }

    /// Read one byte; unwritten anonymous pages read as zero. A [`PageProt::Backed`] page redirects
    /// to its §13 region buffer (so an aliased page reads whatever the shared backing holds).
    fn byte(&self, off: u64) -> u8 {
        let idx = (off % self.page) as usize;
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = self.prot.get(&(off / self.page))
        {
            return self
                .regions
                .get(region)
                .map_or(0, |r| r.read_byte(*region_off + idx as u64));
        }
        self.pages.get(&(off / self.page)).map_or(0, |p| p[idx])
    }

    fn set_byte(&mut self, off: u64, b: u8) {
        let idx = (off % self.page) as usize;
        let page = self.page;
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = self.prot.get(&(off / page))
        {
            // §13 aliased page: write through to the shared region backing.
            if let Some(r) = self.regions.get(region) {
                r.write_byte(*region_off + idx as u64, b);
            }
            return;
        }
        self.pages
            .entry(off / page)
            .or_insert_with(|| vec![0u8; page as usize])[idx] = b;
    }

    /// Seed the low bytes of the window from `init` (escape-oracle, §18). Bytes past the
    /// window size are ignored — confinement only concerns `[0, size)`.
    fn seed(&mut self, init: &[u8]) {
        let n = (init.len() as u64).min(self.window.mapped());
        for i in 0..n {
            self.set_byte(i, init[i as usize]);
        }
    }

    /// Snapshot the low `n` bytes of the window (clamped to the backed `mapped` extent).
    fn snapshot(&self, n: u64) -> Vec<u8> {
        let n = n.min(self.window.mapped());
        (0..n).map(|i| self.byte(i)).collect()
    }

    /// Snapshot the low `min(reserved, max(mapped, snap_cap))` bytes for the escape-oracle —
    /// **including grown reserved-tail pages** (a page absent from the map reads zero, matching the
    /// JIT's freshly-committed tail). Page-wise (one map lookup per committed page, not per byte) so
    /// widening past the backed prefix stays cheap.
    fn snapshot_window(&self, snap_cap: usize) -> Vec<u8> {
        let snap = self
            .window
            .reserved()
            .min(self.window.mapped().max(snap_cap as u64)) as usize;
        let mut out = vec![0u8; snap];
        for (&idx, page) in &self.pages {
            let start = (idx * self.page) as usize;
            if start >= snap {
                continue;
            }
            let n = (self.page as usize).min(snap - start);
            out[start..start + n].copy_from_slice(&page[..n]);
        }
        // §13 aliased pages live in their region backing, not in `pages` — fill them from there.
        for (&idx, p) in &self.prot {
            let PageProt::Backed {
                region, region_off, ..
            } = p
            else {
                continue;
            };
            let start = (idx * self.page) as usize;
            if start >= snap {
                continue;
            }
            let n = (self.page as usize).min(snap - start);
            if let Some(r) = self.regions.get(region) {
                for k in 0..n {
                    out[start + k] = r.read_byte(*region_off + k as u64);
                }
            }
        }
        out
    }
}

impl GuestMem for Mem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.read_bytes_impl(ptr, len)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        self.write_bytes_impl(ptr, data)
    }

    /// §3e op 0 `map`: (re)commit pages with `prot`, zero-filling them (a fresh commit). Works
    /// anywhere in the reserved window `[0, reserved)` — including **growth** into the reserved
    /// tail `[mapped, reserved)`, the §1a sparse-address-space capability. Out-of-range /
    /// misaligned → `-EINVAL`.
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for page in pages {
            self.set_prot(page, prot);
            self.pages.remove(&page); // commit ⇒ fresh zeroed page
        }
        0
    }

    /// §3e op 1 `unmap`: decommit pages — any later access faults, and a re-`map` reads zero.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for page in pages {
            self.prot.insert(page, PageProt::Unmapped);
            self.pages.remove(&page);
        }
        0
    }

    /// §3e op 2 `protect`: change the protection of mapped pages without touching their backing
    /// (the D40 read-only const-segment mechanism: `protect(READ)` ⇒ later stores fault). A §13
    /// aliased page stays aliased — only its writability changes (or it `unmap`s if neither R nor W),
    /// so the shared bytes survive a `protect(READ)`.
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for page in pages {
            if let Some(PageProt::Backed {
                region, region_off, ..
            }) = self.prot.get(&page).copied()
            {
                if prot & (PROT_READ | PROT_WRITE) == 0 {
                    self.prot.insert(page, PageProt::Unmapped);
                } else {
                    self.prot.insert(
                        page,
                        PageProt::Backed {
                            region,
                            region_off,
                            writable: prot & PROT_WRITE != 0,
                        },
                    );
                }
            } else {
                self.set_prot(page, prot);
            }
        }
        0
    }

    /// §13 op 0 `map`: alias `backing`'s `[region_off, region_off+len)` into the window at
    /// `[win_off, win_off+len)`. Both window offsets and the region offset round to whole pages; the
    /// region span must fit the backing; the mapping must be at least readable. The aliased pages'
    /// bytes then live in the region (a prior anonymous page there is dropped), so a store at one
    /// alias is visible at every other mapping of the same region.
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        region: u32,
        backing: RegionBacking,
    ) -> i64 {
        let pages = match self.prot_pages(win_off, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if !region_off.is_multiple_of(self.page) || prot & PROT_READ == 0 {
            return EINVAL;
        }
        match region_off.checked_add(len) {
            Some(end) if end <= backing.size() => {}
            _ => return EINVAL,
        }
        let writable = prot & PROT_WRITE != 0;
        self.regions.insert(region, backing);
        for (i, page) in pages.enumerate() {
            self.prot.insert(
                page,
                PageProt::Backed {
                    region,
                    region_off: region_off + i as u64 * self.page,
                    writable,
                },
            );
            self.pages.remove(&page); // bytes live in the region now, not an anonymous page
        }
        0
    }

    /// §3e op 3 `page_size`: the backing-store page granularity (`self.page`, the host page) — the
    /// unit `map`/`unmap`/`protect` round to. The JIT's `MprotectWindow` reports the same host page,
    /// so the two backends agree.
    fn page_size(&self) -> i64 {
        self.page as i64
    }
}

/// Turn `width` raw little-endian bytes into the loaded value, sign- or zero-
/// extending narrow integer loads into the i32/i64 result type.
fn decode_loaded(rty: ValType, width: u32, signed: bool, raw: u64) -> Value {
    match rty {
        ValType::F32 => Value::F32(f32::from_bits(raw as u32)),
        ValType::F64 => Value::F64(f64::from_bits(raw)),
        ValType::I32 | ValType::I64 => {
            let bits = width * 8;
            let ext = if signed && bits < 64 {
                let shift = 64 - bits;
                (((raw << shift) as i64) >> shift) as u64 // arithmetic sign-extend
            } else {
                raw
            };
            if rty == ValType::I32 {
                Value::I32(ext as i32)
            } else {
                Value::I64(ext as i64)
            }
        }
    }
}

/// The low 64 bits of a value, for storing (the store width selects how many bytes).
fn store_bits(v: Value) -> u64 {
    match v {
        Value::I32(x) => x as u32 as u64,
        Value::I64(x) => x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
    }
}

/// Access width in bytes of an atomic `ty` (§12) — also its natural-alignment requirement.
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Low-`width`-bytes mask (`width` ∈ {4, 8}).
fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Decode the low `ty`-width bytes (zero-extended, as from [`Mem::read_le`]) into a [`Value`].
fn atomic_decode(ty: IntTy, raw: u64) -> Value {
    match ty {
        IntTy::I32 => Value::I32(raw as i32),
        IntTy::I64 => Value::I64(raw as i64),
    }
}

/// Apply an atomic RMW: `old`/`arg` are the low `ty`-width bytes; returns the new low-`width` value.
fn atomic_rmw_apply(ty: IntTy, op: AtomicRmwOp, old: u64, arg: u64) -> u64 {
    match ty {
        IntTy::I32 => {
            let (o, a) = (old as u32, arg as u32);
            let r = match op {
                AtomicRmwOp::Add => o.wrapping_add(a),
                AtomicRmwOp::Sub => o.wrapping_sub(a),
                AtomicRmwOp::And => o & a,
                AtomicRmwOp::Or => o | a,
                AtomicRmwOp::Xor => o ^ a,
                AtomicRmwOp::Xchg => a,
            };
            r as u64
        }
        IntTy::I64 => match op {
            AtomicRmwOp::Add => old.wrapping_add(arg),
            AtomicRmwOp::Sub => old.wrapping_sub(arg),
            AtomicRmwOp::And => old & arg,
            AtomicRmwOp::Or => old | arg,
            AtomicRmwOp::Xor => old ^ arg,
            AtomicRmwOp::Xchg => arg,
        },
    }
}

fn bin32(op: BinOp, a: i32, b: i32) -> Result<i32, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i32::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u32) / (b as u32)) as i32
        }
        BinOp::RemS => {
            // `rem_s` traps only on a zero divisor. `INT_MIN % -1 == 0` — a perfectly
            // representable result, so it does *not* trap: traps are for results with no
            // representable value (§3b), and only the *quotient* overflows here, not the
            // remainder. (`wrapping_rem` yields 0.) See `div_s`, which does trap.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u32) % (b as u32)) as i32
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        // Shift amount is taken mod bitwidth (`wrapping_sh*` masks rhs to 0..31).
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u32).wrapping_shr(b as u32)) as i32,
        // Rotation amount is also mod bitwidth (`rotate_*` reduces it internally).
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

fn intun32(op: IntUnOp, a: i32) -> i32 {
    match op {
        IntUnOp::Clz => (a as u32).leading_zeros() as i32,
        IntUnOp::Ctz => (a as u32).trailing_zeros() as i32,
        IntUnOp::Popcnt => (a as u32).count_ones() as i32,
        IntUnOp::Extend8S => (a as i8) as i32,
        IntUnOp::Extend16S => (a as i16) as i32,
        IntUnOp::Extend32S => a, // identity for i32
    }
}

fn intun64(op: IntUnOp, a: i64) -> i64 {
    match op {
        IntUnOp::Clz => (a as u64).leading_zeros() as i64,
        IntUnOp::Ctz => (a as u64).trailing_zeros() as i64,
        IntUnOp::Popcnt => (a as u64).count_ones() as i64,
        IntUnOp::Extend8S => (a as i8) as i64,
        IntUnOp::Extend16S => (a as i16) as i64,
        IntUnOp::Extend32S => (a as i32) as i64,
    }
}

fn bin64(op: BinOp, a: i64, b: i64) -> Result<i64, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i64::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u64) / (b as u64)) as i64
        }
        BinOp::RemS => {
            // Only a zero divisor traps; `INT_MIN % -1 == 0` is representable (only the
            // quotient overflows, not the remainder), so it returns 0 — see `bin32`.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u64) % (b as u64)) as i64
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u64).wrapping_shr(b as u32)) as i64,
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

#[inline]
fn check_div(by_zero: bool, overflow: bool) -> Result<(), Trap> {
    if by_zero {
        Err(Trap::DivByZero)
    } else if overflow {
        Err(Trap::IntOverflow)
    } else {
        Ok(())
    }
}

fn cmp32(op: CmpOp, a: i32, b: i32) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u32) < (b as u32),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u32) <= (b as u32),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u32) > (b as u32),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u32) >= (b as u32),
    }
}

fn cmp64(op: CmpOp, a: i64, b: i64) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u64) < (b as u64),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u64) <= (b as u64),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u64) > (b as u64),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u64) >= (b as u64),
    }
}

#[inline]
fn step(fuel: &mut u64) -> Result<(), Trap> {
    *fuel = fuel.checked_sub(1).ok_or(Trap::OutOfFuel)?;
    Ok(())
}

#[inline]
fn get(vals: &[Value], v: ValIdx) -> Result<Value, Trap> {
    vals.get(v as usize).copied().ok_or(Trap::Malformed)
}

fn collect(vals: &[Value], idxs: &[ValIdx]) -> Result<Vec<Value>, Trap> {
    idxs.iter().map(|&v| get(vals, v)).collect()
}

#[inline]
/// Encode a value into its `i64` capability-ABI slot (scalars; `i32`/`f32` in the low
/// bits). Mirrors the JIT's marshalling so both drive the same slot-based dispatch.
fn val_to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}

/// Decode a capability-ABI result slot back to a `Value` of the declared type.
fn slot_to_val(ty: ValType, s: i64) -> Value {
    match ty {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
    }
}

fn as_i32(v: Value) -> Result<i32, Trap> {
    match v {
        Value::I32(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_i64(v: Value) -> Result<i64, Trap> {
    match v {
        Value::I64(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_f32(v: Value) -> Result<f32, Trap> {
    match v {
        Value::F32(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[inline]
fn as_f64(v: Value) -> Result<f64, Trap> {
    match v {
        Value::F64(x) => Ok(x),
        _ => Err(Trap::Malformed),
    }
}

#[cfg(test)]
mod prot_tests {
    //! White-box tests for the guest-visible page-protection model (`map`/`unmap`/`protect`,
    //! §3e Memory cap / §4) — the reference semantics the JIT's `mprotect`-backed side is
    //! differential-tested against next. Granularity is the **host** page size (4 KiB / 16 KiB),
    //! same as `Mem`, so these pass on any host.
    use super::*;

    /// The host page size — the protection granularity these tests align to.
    fn page() -> u64 {
        host_page_size()
    }

    /// A fully-mapped 64 KiB window (`mapped == reserved`, 16 pages).
    fn mem64k() -> Mem {
        Mem::with_reservation(0, 16)
    }

    #[test]
    fn protect_read_only_faults_store_allows_load() {
        let mut m = mem64k();
        let v = Value::I64(0x1122_3344_5566_7788u64 as i64);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        // a store to the RO page faults; the value is still readable
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        // an adjacent, unprotected page is unaffected
        assert!(m.store(page(), 0, StoreOp::I64, Value::I64(7)).is_ok());
    }

    #[test]
    fn protect_rw_restores_writability() {
        let mut m = mem64k();
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.protect(0, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unmap_faults_then_remap_zeroes() {
        let mut m = mem64k();
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x42)).is_ok());
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // re-commit ⇒ accessible again and zeroed
        assert_eq!(m.map(0, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn bad_args_einval() {
        let mut m = mem64k();
        assert_eq!(m.protect(1, page(), PROT_READ), EINVAL); // misaligned offset
        assert_eq!(m.protect(0, 0, PROT_READ), EINVAL); // zero length
                                                        // mem64k is fully mapped (reserved == mapped == 64 KiB), so its tail is empty: a range
                                                        // at/past the reserved top is still out of range.
        assert_eq!(m.unmap(65536, page()), EINVAL); // offset == reserved ⇒ out of range
        assert_eq!(m.map(0, 1 << 20, PROT_WRITE), EINVAL); // len past reserved
    }

    /// A window whose reserved mask domain (`1 MiB`) is larger than the initial backed prefix
    /// (`64 KiB`): the tail `[64 KiB, 1 MiB)` is reserved-but-unmapped and the guest can grow into
    /// it. `Mem::with_reservation(reserved_log2=20, mapped_log2=16)`.
    fn mem_growable() -> Mem {
        Mem::with_reservation(20, 16)
    }

    #[test]
    fn tail_access_faults_until_mapped() {
        let mut m = mem_growable();
        let tail = 1u64 << 16; // first byte of the reserved tail (64 KiB)
                               // Untouched tail faults (any access) — it is reserved-but-unmapped.
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // Grow one page into the tail; now it is committed, zeroed, read-write.
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(0x99)).is_ok());
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0x99)));
        // The next page up is still unmapped.
        assert_eq!(
            m.load(tail + page(), 0, LoadOp::I64),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn grow_then_unmap_faults_again() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(7)).is_ok());
        assert_eq!(m.unmap(tail, page()), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        // Re-mapping zero-fills (the old contents are gone).
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
    }

    #[test]
    fn grow_read_only_then_store_faults() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        // Map a tail page read-only: reads of the (zeroed) page succeed, a store faults.
        assert_eq!(m.map(tail, page(), PROT_READ), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn growth_bounds_are_reserved_not_mapped() {
        let mut m = mem_growable();
        let reserved = 1u64 << 20;
        // Mapping anywhere in the reserved tail is allowed now (was EINVAL pre-growth).
        assert_eq!(m.map(1 << 16, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.map(reserved - page(), page(), PROT_READ | PROT_WRITE), 0);
        // At/past the reserved top is still out of range.
        assert_eq!(m.map(reserved, page(), PROT_WRITE), EINVAL);
        assert_eq!(m.unmap(reserved - page(), 2 * page()), EINVAL);
    }

    #[test]
    fn grown_tail_buffer_borrow_round_trips() {
        // A cap buffer (§7 borrow) in a grown tail region validates and round-trips; one in the
        // unmapped tail is rejected (-EFAULT ⇒ None).
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_none()); // unmapped ⇒ EFAULT
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_some());
        assert_eq!(m.read_bytes_impl(tail, 4), Some(vec![1, 2, 3, 4]));
        // A borrow straddling the committed/uncommitted page boundary is rejected.
        assert!(m.read_bytes_impl(tail + page() - 2, 4).is_none());
    }

    #[test]
    fn cross_page_store_faults_if_either_page_protected() {
        let mut m = mem64k();
        // page 1 read-only; an 8-byte store straddling the page-0/1 boundary touches page 1.
        assert_eq!(m.protect(page(), page(), PROT_READ), 0);
        assert_eq!(
            m.store(page() - 4, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // fully within page 0 (still rw) is fine
        assert!(m.store(page() - 8, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unprotected_window_is_unrestricted() {
        // With an empty protection map, check_prot is a no-op: every in-window access works.
        let mut m = mem64k();
        for off in [0u64, 8, page(), 65536 - 8] {
            assert!(m.store(off, 0, StoreOp::I64, Value::I64(0x55)).is_ok());
            assert_eq!(m.load(off, 0, LoadOp::I64), Ok(Value::I64(0x55)));
        }
    }

    // ---- §13 SharedRegion: host-backed memory aliased into the window ----

    /// A §13 `SharedRegion` backing of `pages` whole host pages, zero-filled.
    fn region(pages: u64) -> RegionBacking {
        Rc::new(VecBacking(RefCell::new(vec![
            0u8;
            (pages * page()) as usize
        ])))
    }

    #[test]
    fn shared_region_aliases_two_window_offsets() {
        // One region mapped at two window offsets names the *same* bytes: a store at one alias is
        // visible at the other (and vice versa) — the §13 zero-overhead aliasing primitive.
        let mut m = mem64k();
        let r = region(1);
        let (a, b) = (0, page());
        assert_eq!(
            m.map_region(a, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(b, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        let v = Value::I64(0x0123_4567_89ab_cdefu64 as i64);
        assert!(m.store(a, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.load(b, 0, LoadOp::I64), Ok(v), "A→B alias");
        let w = Value::I64(0x7777);
        assert!(m.store(b + 16, 0, StoreOp::I64, w).is_ok());
        assert_eq!(m.load(a + 16, 0, LoadOp::I64), Ok(w), "B→A alias");
    }

    #[test]
    fn shared_region_offsets_are_region_relative() {
        // Pointers are region-relative (§13): the same *region* offset at two window offsets aliases;
        // different region offsets are independent.
        let mut m = mem64k();
        let r = region(2);
        // window pages 0,1 ⇒ region pages 0,1.
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        // a second mapping of *region page 1* at window page 2.
        assert_eq!(
            m.map_region(2 * page(), page(), page(), PROT_READ | PROT_WRITE, 0, r),
            0
        );
        let v = Value::I64(0xdead_beef);
        assert!(m.store(page(), 0, StoreOp::I64, v).is_ok()); // write region page 1 via window page 1
        assert_eq!(m.load(2 * page(), 0, LoadOp::I64), Ok(v)); // observe via window page 2
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0))); // region page 0 independent
    }

    #[test]
    fn shared_region_read_only_alias_shares_reads_faults_stores() {
        let mut m = mem64k();
        let r = region(1);
        assert_eq!(
            m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(page(), 0, page(), PROT_READ, 0, r), 0); // RO alias of same region
        let v = Value::I64(0x5151_5151);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(
            m.load(page(), 0, LoadOp::I64),
            Ok(v),
            "RO alias sees the write"
        );
        assert_eq!(
            m.store(page(), 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault),
            "store to RO alias faults"
        );
        // protect(READ) on the RW alias keeps it aliased (shared bytes survive), now store-faulting.
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(2)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn shared_region_unmap_drops_alias_and_map_replaces_anonymous() {
        let mut m = mem64k();
        // Aliasing over an already-written anonymous page redirects to the region (old bytes gone).
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x4242)).is_ok());
        let r = region(1);
        assert_eq!(m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        assert_eq!(
            m.load(0, 0, LoadOp::I64),
            Ok(Value::I64(0)),
            "region zero-fill"
        );
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(9)).is_ok());
        // unmap drops the alias → faults.
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
    }

    #[test]
    fn shared_region_bad_args_einval() {
        let mut m = mem64k();
        let r = region(1); // one page
        assert_eq!(m.map_region(1, 0, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned window
        assert_eq!(m.map_region(0, 0, 0, PROT_READ, 0, r.clone()), EINVAL); // zero len
        assert_eq!(m.map_region(0, 1, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned region
        assert_eq!(
            m.map_region(0, page(), page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // region OOB
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // span > backing
        assert_eq!(m.map_region(0, 0, page(), PROT_WRITE, 0, r.clone()), EINVAL); // not readable
        assert_eq!(m.map_region(65536, 0, page(), PROT_READ, 0, r), EINVAL); // window past reserved
    }
}
