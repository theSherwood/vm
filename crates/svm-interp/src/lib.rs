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
    /// Signed `INT_MIN / -1` (or `rem_s`) overflow (§3b).
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
    let f = m.funcs.get(func as usize).ok_or(Trap::Malformed)?;
    // One linear-memory window per run, zero-initialized and lazily paged. The whole
    // module shares it (all functions address the same window).
    let mut mem = m.memory.map(|mc| Mem::new(mc.size_log2));
    run_func(f, args, fuel, &mut mem, &m.funcs, 0)
}

fn run_func<'a>(
    mut f: &'a Func,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    funcs: &'a [Func],
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
                    let results = run_func(callee, &argv, fuel, mem, funcs, depth + 1)?;
                    vals.extend(results);
                } else if let Inst::CallIndirect { ty, idx, args } = inst {
                    let callee = table_lookup(funcs, as_i32(get(&vals, *idx)?)?, ty)?;
                    let argv = collect(&vals, args)?;
                    let results = run_func(callee, &argv, fuel, mem, funcs, depth + 1)?;
                    vals.extend(results);
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
        Inst::IToFConv { op, a } => i_to_f(*op, get(vals, *a)?)?,
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
        Inst::Store { .. } | Inst::Call { .. } | Inst::CallIndirect { .. } => return Ok(None),
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
// Linear memory — the confinement-masking *reference* (§4, invariant I1)
// ----------------------------------------------------------------------------

/// Page size for the lazy backing store. Lazy paging means interpreter memory is
/// bounded by what a (fuel-limited) run actually touches, so even a huge declared
/// window (e.g. `size_log2 = 63`) never eagerly allocates — safe to fuzz.
const PAGE: u64 = 4096;

/// A guest linear-memory window. The whole point is the **confinement invariant**:
/// every access is masked to `[0, size)` with `addr & (size − 1)` (size is a power
/// of two), and an access that would cross the top of the window faults (modeling
/// the §4 guard region). This is the semantics the JIT masking lowering is
/// differential-tested against (§18).
struct Mem {
    size: u64,
    mask: u64,
    pages: BTreeMap<u64, Vec<u8>>,
}

impl Mem {
    fn new(size_log2: u8) -> Mem {
        // `size_log2 < 64` for verified modules; clamp defensively so an unverified
        // (fuzzed) module can never trigger a shift overflow panic.
        let size = 1u64 << size_log2.min(63);
        Mem {
            size,
            mask: size - 1,
            pages: BTreeMap::new(),
        }
    }

    /// The confined final effective offset: mask of `addr + offset` into `[0, size)`.
    /// Masking the *final* address (after folding the immediate offset) is the
    /// load-bearing security property — see §4.
    fn confine(&self, addr: u64, offset: u64) -> u64 {
        addr.wrapping_add(offset) & self.mask
    }

    /// Confine, then guard-check that the whole `width`-byte access stays in-window.
    /// A boundary-crossing access faults (the guard region, §4).
    fn range(&self, addr: u64, offset: u64, width: u32) -> Result<u64, Trap> {
        let base = self.confine(addr, offset);
        match base.checked_add(width as u64) {
            Some(end) if end <= self.size => Ok(base),
            _ => Err(Trap::MemoryFault),
        }
    }

    fn load(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Value, Trap> {
        let (_, rty, width, signed) = op.info();
        let base = self.range(addr, offset, width)?;
        let raw = self.read_le(base, width);
        Ok(decode_loaded(rty, width, signed, raw))
    }

    fn store(&mut self, addr: u64, offset: u64, op: StoreOp, v: Value) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        let base = self.range(addr, offset, width)?;
        // `write_le` keeps only the low `width` bytes, so narrow stores truncate.
        self.write_le(base, width, store_bits(v));
        Ok(())
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
            check_div(b == 0, a == i32::MIN && b == -1)?;
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
            check_div(b == 0, a == i64::MIN && b == -1)?;
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
