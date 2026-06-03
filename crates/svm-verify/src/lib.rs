//! The verifier — security-critical TCB (`DESIGN.md` §2a, invariants I2/I3/I4;
//! §3b "verifier validity rules").
//!
//! A single linear forward pass, O(module size), no dominance analysis and no
//! fixups (block parameters make cross-block dataflow explicit). For each block we
//! seed a local type vector with the block's declared parameter types, walk the
//! instructions (checking each operand is defined *earlier* and exactly the right
//! type, then appending the result type), and finally check the terminator's branch
//! arguments against each target block's declared parameter types.
//!
//! **Fail-closed:** any violation returns `Err`; the verifier never panics on any
//! input (that property is fuzzed — see the `svm` crate). A module that verifies is
//! the precondition for the escape-freedom contract (§2a); soundness of *this code*
//! is the separate hard problem (§18).
#![forbid(unsafe_code)]

use svm_ir::{BlockIdx, Func, Inst, Module, Terminator, ValIdx, ValType};

/// Why verification rejected a module. Carries enough location to debug, never
/// enough to be load-bearing for safety (the boolean accept/reject is the contract).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VerifyError {
    /// Entry block parameter types must equal the function signature's parameters.
    EntryParamsMismatch { func: u32 },
    /// A branch/return references a block index that does not exist.
    BlockOutOfRange { func: u32, block: u32, target: u32 },
    /// An operand references a value index not yet defined in this block.
    ValueOutOfRange {
        func: u32,
        block: u32,
        value: ValIdx,
        defined: u32,
    },
    /// An operand had the wrong type for its opcode.
    TypeMismatch {
        func: u32,
        block: u32,
        expected: ValType,
        found: ValType,
    },
    /// Branch argument count did not match the target block's parameter count.
    ArgCountMismatch {
        func: u32,
        block: u32,
        target: u32,
        expected: usize,
        found: usize,
    },
    /// Return value count did not match the function's result count.
    ResultCountMismatch {
        func: u32,
        block: u32,
        expected: usize,
        found: usize,
    },
}

/// Verify an entire module. `Ok(())` is the only "accept".
pub fn verify_module(m: &Module) -> Result<(), VerifyError> {
    for (fi, f) in m.funcs.iter().enumerate() {
        verify_func(fi as u32, f)?;
    }
    Ok(())
}

fn verify_func(fi: u32, f: &Func) -> Result<(), VerifyError> {
    // Per function: the entry block's parameters are the function's parameters.
    match f.blocks.first() {
        Some(entry) if entry.params == f.params => {}
        Some(_) => return Err(VerifyError::EntryParamsMismatch { func: fi }),
        // A function with no blocks cannot return; treat as ill-formed.
        None => return Err(VerifyError::EntryParamsMismatch { func: fi }),
    }

    let nblocks = f.blocks.len() as u32;
    for (bi, b) in f.blocks.iter().enumerate() {
        let bi = bi as u32;
        // Seed the local type vector with the block's declared parameter types.
        let mut types: Vec<ValType> = b.params.clone();

        for inst in &b.insts {
            let result = check_inst(fi, bi, inst, &types)?;
            types.push(result);
        }

        check_terminator(fi, bi, &b.term, &types, nblocks, f)?;
    }
    Ok(())
}

/// Check one instruction's operands against the running type vector and return the
/// result type to append. Operands must reference strictly-earlier indices.
fn check_inst(fi: u32, bi: u32, inst: &Inst, types: &[ValType]) -> Result<ValType, VerifyError> {
    match inst {
        Inst::I32Const(_) | Inst::I64Const(_) => {}
        Inst::I32Add(a, b) => {
            expect(fi, bi, types, *a, ValType::I32)?;
            expect(fi, bi, types, *b, ValType::I32)?;
        }
        Inst::I64Add(a, b) => {
            expect(fi, bi, types, *a, ValType::I64)?;
            expect(fi, bi, types, *b, ValType::I64)?;
        }
    }
    Ok(inst.result_type())
}

fn check_terminator(
    fi: u32,
    bi: u32,
    term: &Terminator,
    types: &[ValType],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    match term {
        Terminator::Br { target, args } => {
            check_branch(fi, bi, *target, args, types, nblocks, f)?;
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            expect(fi, bi, types, *cond, ValType::I32)?;
            check_branch(fi, bi, *then_blk, then_args, types, nblocks, f)?;
            check_branch(fi, bi, *else_blk, else_args, types, nblocks, f)?;
        }
        Terminator::Return(vals) => {
            if vals.len() != f.results.len() {
                return Err(VerifyError::ResultCountMismatch {
                    func: fi,
                    block: bi,
                    expected: f.results.len(),
                    found: vals.len(),
                });
            }
            for (v, want) in vals.iter().zip(&f.results) {
                expect(fi, bi, types, *v, *want)?;
            }
        }
    }
    Ok(())
}

/// Check a single branch edge: target in range, arg count + types match the target
/// block's declared parameters exactly.
fn check_branch(
    fi: u32,
    bi: u32,
    target: BlockIdx,
    args: &[ValIdx],
    types: &[ValType],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    if target >= nblocks {
        return Err(VerifyError::BlockOutOfRange {
            func: fi,
            block: bi,
            target,
        });
    }
    let target_params = &f.blocks[target as usize].params;
    if args.len() != target_params.len() {
        return Err(VerifyError::ArgCountMismatch {
            func: fi,
            block: bi,
            target,
            expected: target_params.len(),
            found: args.len(),
        });
    }
    for (v, want) in args.iter().zip(target_params) {
        expect(fi, bi, types, *v, *want)?;
    }
    Ok(())
}

/// An operand must be defined earlier in this block and have exactly `want`'s type.
fn expect(
    fi: u32,
    bi: u32,
    types: &[ValType],
    v: ValIdx,
    want: ValType,
) -> Result<(), VerifyError> {
    let found = types
        .get(v as usize)
        .copied()
        .ok_or(VerifyError::ValueOutOfRange {
            func: fi,
            block: bi,
            value: v,
            defined: types.len() as u32,
        })?;
    if found != want {
        return Err(VerifyError::TypeMismatch {
            func: fi,
            block: bi,
            expected: want,
            found,
        });
    }
    Ok(())
}
