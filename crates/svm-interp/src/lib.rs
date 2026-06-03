//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

use svm_ir::{Func, FuncIdx, Inst, Module, Terminator, ValIdx};

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
                let next = collect(&vals, args)?;
                block_idx = *target as usize;
                vals = next;
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let taken = match get(&vals, *cond)? {
                    Value::I32(c) => c != 0,
                    _ => return Err(Trap::Malformed),
                };
                let (target, edge_args) = if taken {
                    (*then_blk, then_args)
                } else {
                    (*else_blk, else_args)
                };
                let next = collect(&vals, edge_args)?;
                block_idx = target as usize;
                vals = next;
            }
            Terminator::Return(out) => return collect(&vals, out),
        }
    }
}

fn eval_inst(inst: &Inst, vals: &[Value]) -> Result<Value, Trap> {
    Ok(match inst {
        Inst::I32Const(c) => Value::I32(*c),
        Inst::I64Const(c) => Value::I64(*c),
        Inst::I32Add(a, b) => {
            let a = as_i32(get(vals, *a)?)?;
            let b = as_i32(get(vals, *b)?)?;
            // Two's-complement wrap (§3b: integer add wraps, no trap).
            Value::I32(a.wrapping_add(b))
        }
        Inst::I64Add(a, b) => {
            let a = as_i64(get(vals, *a)?)?;
            let b = as_i64(get(vals, *b)?)?;
            Value::I64(a.wrapping_add(b))
        }
    })
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
