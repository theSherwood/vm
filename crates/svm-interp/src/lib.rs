//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use svm_ir::{
    BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx, FuncType,
    IToF, Inst, IntTy, IntUnOp, LoadOp, Module, StoreOp, Terminator, ValIdx, ValType,
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

/// Maximum nested `call` depth before the interpreter traps. This bounds the host
/// stack the recursive interpreter consumes, so adversarial (or merely deep) guest
/// recursion yields a clean `Trap::StackOverflow` instead of crashing the process.
///
/// Kept conservative because each frame must fit even a small (≈2 MiB) thread stack
/// — this is a reference-oracle limit, not the production recursion ceiling (the JIT
/// uses the guest's guard-paged data stack, §5, not host recursion).
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
    // One linear-memory window per run, zero-initialized and lazily paged. The whole
    // module shares it (all functions address the same window).
    let mut mem = m.memory.map(|mc| Mem::new(mc.size_log2));
    run_func(f, args, fuel, &mut mem, &m.funcs, host, 0)
}

fn run_func<'a>(
    mut f: &'a Func,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    funcs: &'a [Func],
    host: &mut Host,
    depth: u32,
) -> Result<Vec<Value>, Trap> {
    if depth > MAX_CALL_DEPTH {
        return Err(Trap::StackOverflow);
    }
    // Entry block parameters are the function arguments (verifier guarantees the
    // count/types; we still copy defensively).
    let mut vals: Vec<Value> = args.to_vec();

    // Outer loop: a *tail* call replaces the current function in place — same host
    // frame, no depth growth — restarting here with the callee and its args. So
    // tail-recursive guests run in O(1) host stack (bounded only by fuel).
    'tail: loop {
        let mut block_idx: usize = 0;

        loop {
            let block = f.blocks.get(block_idx).ok_or(Trap::Malformed)?;

            for inst in &block.insts {
                step(fuel)?;
                // Non-tail calls recurse (they append 0..N results and continue);
                // the rest go through `eval_inst` (one value, or none for `Store`).
                if let Inst::Call { func, args } = inst {
                    let argv = collect(&vals, args)?;
                    let callee = funcs.get(*func as usize).ok_or(Trap::Malformed)?;
                    let results = run_func(callee, &argv, fuel, mem, funcs, host, depth + 1)?;
                    vals.extend(results);
                } else if let Inst::CallIndirect { ty, idx, args } = inst {
                    let callee = table_lookup(funcs, as_i32(get(&vals, *idx)?)?, ty)?;
                    let argv = collect(&vals, args)?;
                    let results = run_func(callee, &argv, fuel, mem, funcs, host, depth + 1)?;
                    vals.extend(results);
                } else if let Inst::CapCall {
                    type_id,
                    op,
                    sig,
                    handle,
                    args,
                } = inst
                {
                    // Capability call (§3c): resolve the handle in the host-owned table
                    // (mask + type_id/generation check) and dispatch to the mock host.
                    // Args/results cross as i64 slots (the shared host-dispatch ABI).
                    let h = as_i32(get(&vals, *handle)?)?;
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(val_to_slot(get(&vals, *a)?));
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let results = host.cap_dispatch_slots(*type_id, *op, h, &argv, gm)?;
                    for (s, ty) in results.iter().zip(&sig.results) {
                        vals.push(slot_to_val(*ty, *s));
                    }
                } else if let Some(v) = eval_inst(inst, &vals, mem)? {
                    vals.push(v);
                }
            }

            step(fuel)?;
            match &block.term {
                Terminator::Br { target, args } => {
                    vals = collect(&vals, args)?;
                    block_idx = *target as usize;
                }
                Terminator::BrIf {
                    cond,
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                } => {
                    let (target, edge_args) = if as_i32(get(&vals, *cond)?)? != 0 {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    vals = collect(&vals, edge_args)?;
                    block_idx = target as usize;
                }
                Terminator::BrTable {
                    idx,
                    targets,
                    default,
                } => {
                    let i = as_i32(get(&vals, *idx)?)? as u32 as usize;
                    let (target, edge_args) = targets.get(i).unwrap_or(default);
                    vals = collect(&vals, edge_args)?;
                    block_idx = *target as usize;
                }
                Terminator::Return(out) => return collect(&vals, out),
                Terminator::Unreachable => return Err(Trap::Unreachable),
                Terminator::ReturnCall { func, args } => {
                    let argv = collect(&vals, args)?;
                    f = funcs.get(*func as usize).ok_or(Trap::Malformed)?;
                    vals = argv;
                    continue 'tail;
                }
                Terminator::ReturnCallIndirect { ty, idx, args } => {
                    let callee = table_lookup(funcs, as_i32(get(&vals, *idx)?)?, ty)?;
                    let argv = collect(&vals, args)?;
                    f = callee;
                    vals = argv;
                    continue 'tail;
                }
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
        // Handled before/around the match; listed for exhaustiveness (no panic).
        Inst::Store { .. }
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
    /// `Memory` — op 0 `map`, 1 `unmap`, 2 `protect` (§3e; eager-mapped → no-ops).
    pub const MEMORY: u32 = 3;
}

/// Negative-errno values returned by capability ops (§3e D42): `< 0` is `-errno`,
/// `>= 0` is success. Errors do **not** trap — traps stay reserved for escape/fatal.
const EFAULT: i64 = -14; // buffer not fully within the window
const EINVAL: i64 = -22; // bad op / argument

/// The guest window a capability handler borrows `(ptr, len)` buffers from (§7). Both
/// the interpreter's lazily-paged [`Mem`] and a JIT's flat window implement this, so a
/// single host dispatch ([`Host::cap_dispatch`]) serves both backends. The methods
/// bounds-check `[ptr, ptr+len) ⊆ [0, size)` and return `None` (→ `-EFAULT`) otherwise.
pub trait GuestMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>>;
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()>;
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
                // map/unmap/protect: the window is eagerly mapped, so these succeed
                // as no-ops (0); an unknown op is -EINVAL.
                Ok(vec![if op <= 2 { 0 } else { EINVAL }])
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

/// Page size for the lazy backing store. Lazy paging means interpreter memory is
/// bounded by what a (fuel-limited) run actually touches, so even a huge declared
/// window (e.g. `size_log2 = 63`) never eagerly allocates — safe to fuzz.
const PAGE: u64 = 4096;

/// A guest linear-memory window. Confinement itself lives in [`svm_mask::Window`]
/// (the isolated, separately-fuzzed security unit, §4); `Mem` just owns the lazily
/// paged backing store and threads accesses through that confinement. This is the
/// semantics the JIT masking lowering is differential-tested against (§18).
struct Mem {
    window: Window,
    pages: BTreeMap<u64, Vec<u8>>,
}

impl Mem {
    fn new(size_log2: u8) -> Mem {
        Mem {
            window: Window::new(size_log2),
            pages: BTreeMap::new(),
        }
    }

    fn load(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Value, Trap> {
        let (_, rty, width, signed) = op.info();
        // Confine the access (mask + guard check); a window-crossing access faults.
        let base = self
            .window
            .checked(addr, offset, width)
            .ok_or(Trap::MemoryFault)?;
        let raw = self.read_le(base, width);
        Ok(decode_loaded(rty, width, signed, raw))
    }

    fn store(&mut self, addr: u64, offset: u64, op: StoreOp, v: Value) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        let base = self
            .window
            .checked(addr, offset, width)
            .ok_or(Trap::MemoryFault)?;
        // `write_le` keeps only the low `width` bytes, so narrow stores truncate.
        self.write_le(base, width, store_bits(v));
        Ok(())
    }

    /// Borrow-validate and read a `(ptr, len)` capability buffer (§7): `[ptr, ptr+len)`
    /// must lie fully within the window. Returns the bytes, or `None` (→ `-EFAULT`).
    /// Confinement holds regardless; this explicit check is the recoverable guest-bug
    /// path, not a safety boundary.
    fn read_bytes_impl(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        if end > self.window.size() {
            return None;
        }
        Some((0..len).map(|k| self.byte(ptr + k)).collect())
    }

    /// Borrow-validate and write a `(ptr, len)` capability buffer (§7). `None` → `-EFAULT`.
    fn write_bytes_impl(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let end = ptr.checked_add(data.len() as u64)?;
        if end > self.window.size() {
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

    /// Read one byte; unwritten pages read as zero.
    fn byte(&self, off: u64) -> u8 {
        let idx = (off % PAGE) as usize;
        self.pages.get(&(off / PAGE)).map_or(0, |p| p[idx])
    }

    fn set_byte(&mut self, off: u64, b: u8) {
        let idx = (off % PAGE) as usize;
        self.pages
            .entry(off / PAGE)
            .or_insert_with(|| vec![0u8; PAGE as usize])[idx] = b;
    }
}

impl GuestMem for Mem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.read_bytes_impl(ptr, len)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        self.write_bytes_impl(ptr, data)
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
