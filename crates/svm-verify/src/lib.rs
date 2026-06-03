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
//! Result types are computed here from opcode + operand types (§3a "inferred result
//! types"); for the one polymorphic op (`select`) the result is the operand type.
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
    let cx = Cx { fi, bi, types };
    Ok(match inst {
        Inst::ConstI32(_) => ValType::I32,
        Inst::ConstI64(_) => ValType::I64,
        Inst::IntBin { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::IntCmp { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            ValType::I32
        }
        Inst::Eqz { ty, a } => {
            cx.expect(*a, ty.val())?;
            ValType::I32
        }
        Inst::Convert { op, a } => {
            let (_, src, dst) = op.sig();
            cx.expect(*a, src)?;
            dst
        }
        Inst::Select { cond, a, b } => {
            cx.expect(*cond, ValType::I32)?;
            // Polymorphic: `a` defines the result type, `b` must match it.
            let t = cx.type_of(*a)?;
            cx.expect(*b, t)?;
            t
        }
    })
}

fn check_terminator(
    fi: u32,
    bi: u32,
    term: &Terminator,
    types: &[ValType],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    let cx = Cx { fi, bi, types };
    match term {
        Terminator::Br { target, args } => {
            check_edge(&cx, *target, args, nblocks, f)?;
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            cx.expect(*cond, ValType::I32)?;
            check_edge(&cx, *then_blk, then_args, nblocks, f)?;
            check_edge(&cx, *else_blk, else_args, nblocks, f)?;
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            cx.expect(*idx, ValType::I32)?;
            for (t, args) in targets {
                check_edge(&cx, *t, args, nblocks, f)?;
            }
            let (t, args) = default;
            check_edge(&cx, *t, args, nblocks, f)?;
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
                cx.expect(*v, *want)?;
            }
        }
    }
    Ok(())
}

/// Check a single branch edge: target in range, arg count + types match the target
/// block's declared parameters exactly.
fn check_edge(
    cx: &Cx,
    target: BlockIdx,
    args: &[ValIdx],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    if target >= nblocks {
        return Err(VerifyError::BlockOutOfRange {
            func: cx.fi,
            block: cx.bi,
            target,
        });
    }
    let target_params = &f.blocks[target as usize].params;
    if args.len() != target_params.len() {
        return Err(VerifyError::ArgCountMismatch {
            func: cx.fi,
            block: cx.bi,
            target,
            expected: target_params.len(),
            found: args.len(),
        });
    }
    for (v, want) in args.iter().zip(target_params) {
        cx.expect(*v, *want)?;
    }
    Ok(())
}

/// Bundles the location + running type vector for concise operand checks.
struct Cx<'a> {
    fi: u32,
    bi: u32,
    types: &'a [ValType],
}

impl Cx<'_> {
    /// The type of an earlier-defined operand, or `ValueOutOfRange`.
    fn type_of(&self, v: ValIdx) -> Result<ValType, VerifyError> {
        self.types
            .get(v as usize)
            .copied()
            .ok_or(VerifyError::ValueOutOfRange {
                func: self.fi,
                block: self.bi,
                value: v,
                defined: self.types.len() as u32,
            })
    }

    /// An operand must be defined earlier in this block and have exactly `want`'s type.
    fn expect(&self, v: ValIdx, want: ValType) -> Result<(), VerifyError> {
        let found = self.type_of(v)?;
        if found != want {
            return Err(VerifyError::TypeMismatch {
                func: self.fi,
                block: self.bi,
                expected: want,
                found,
            });
        }
        Ok(())
    }
}
