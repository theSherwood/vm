//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

use svm_ir::{
    BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx, IToF, Inst,
    IntTy, Module, Terminator, ValIdx,
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
    /// Structurally invalid in a way a verified module never is (defensive only).
    Malformed,
}

/// Run `func` with `args`, consuming up to `*fuel` execution steps.
///
/// Returns the function's result values, or a `Trap`. Decrements `*fuel` per
/// instruction and per branch so that even an infinite loop terminates — important
/// for fuzzing and for never hanging a test.
pub fn run(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    let f = m.funcs.get(func as usize).ok_or(Trap::Malformed)?;
    run_func(f, args, fuel)
}

fn run_func(f: &Func, args: &[Value], fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    // Block-local value vector: parameters first, then instruction results.
    let mut block_idx: usize = 0;
    // Entry block parameters are the function arguments (verifier guarantees the
    // count/types; we still copy defensively).
    let mut vals: Vec<Value> = args.to_vec();

    loop {
        let block = f.blocks.get(block_idx).ok_or(Trap::Malformed)?;

        for inst in &block.insts {
            step(fuel)?;
            let v = eval_inst(inst, &vals)?;
            vals.push(v);
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
        }
    }
}

fn eval_inst(inst: &Inst, vals: &[Value]) -> Result<Value, Trap> {
    Ok(match inst {
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
    })
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
    })
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
