//! LLVM-bitcode → SVM-IR translator (the AOT LLVM on-ramp, D54). See `LLVM.md` for the
//! design, the decisions (binding, legalization), and the roadmap.
//!
//! **Trust:** this is an *untrusted frontend* (§2a). Everything it emits is re-checked by
//! `svm-verify`, so a translation bug is a clean error, never an escape. Correctness here is
//! a capability concern, not a safety one.
//!
//! **Pipeline (LLVM.md §4):** legalization is done *out of process* — `clang -O2 -emit-llvm`
//! runs `mem2reg`/SROA so scalars arrive in SSA registers (the §3a two-stack split for free)
//! and only address-taken `alloca`s remain. This crate ingests the legalized bitcode read-only
//! and walks it; it never runs an in-process pass manager.
//!
//! **Scope (Milestone 0):** the absolute floor — a single-basic-block function over `i32`/`i64`
//! integer arithmetic, terminated by `ret`/`unreachable`. Everything outside this frozen subset
//! is a fail-closed [`Error::Unsupported`] (the `unsup` chokepoint, mirroring `svm-wasm`). The
//! scalar+memory+call MVP (φ→block-params, GEP, loads/stores, calls) lands in Milestone 1.

use std::collections::HashMap;
use std::path::Path;

use llvm_ir::constant::Constant;
use llvm_ir::instruction::Instruction;
use llvm_ir::terminator::Terminator as LTerm;
use llvm_ir::types::Type;
use llvm_ir::{Module as LModule, Name, Operand};

use svm_ir::{BinOp, Block, Func, Inst, IntTy, Module, Terminator, ValIdx, ValType};

/// Why a translation could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A construct outside the frozen MVP subset. Fail-closed by design (LLVM.md §2/§8):
    /// we never emit IR we can't stand behind. Widen the subset, never silently mis-translate.
    Unsupported(String),
    /// libLLVM could not parse the bitcode (e.g. produced by an off-version LLVM — we pin 18).
    Parse(String),
}

/// Shorthand for the fail-closed chokepoint (the `svm-wasm` `unsup(...)` analog).
fn unsup<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported(what.into()))
}

/// Translate a legalized LLVM bitcode file (`*.bc`) into a verifier-checkable [`Module`].
/// The bitcode must come from the pinned LLVM (18); off-version input is an [`Error::Parse`].
pub fn translate_bc_path(path: impl AsRef<Path>) -> Result<Module, Error> {
    let m = LModule::from_bc_path(path).map_err(Error::Parse)?;
    translate(&m)
}

/// Translate an already-parsed `llvm-ir` module.
pub fn translate(m: &LModule) -> Result<Module, Error> {
    let mut funcs = Vec::with_capacity(m.functions.len());
    for f in &m.functions {
        funcs.push(translate_func(f)?);
    }
    Ok(Module {
        funcs,
        memory: None,
        data: Vec::new(),
    })
}

/// Map an LLVM type to an SVM value type. Narrow integers collapse to `i32` (§3b: `i8`/`i16`
/// are memory widths only, not SSA value types); `i64` stays `i64`. Wider/other types are
/// outside the Milestone-0 subset.
fn val_type(ty: &Type) -> Result<ValType, Error> {
    match ty {
        Type::IntegerType { bits } if *bits <= 32 => Ok(ValType::I32),
        Type::IntegerType { bits } if *bits == 64 => Ok(ValType::I64),
        Type::IntegerType { bits } => unsup(format!("integer width i{bits} (Milestone 1+)")),
        other => unsup(format!("type {other} (Milestone 1+)")),
    }
}

/// The `IntTy` an integer operand is computed at (drives the typed `iN.*` op). Read straight
/// off the operand — a local carries its `TypeRef`, a constant its bit width — so we never need
/// an `llvm-ir` type-inference context.
fn operand_int_ty(op: &Operand) -> Result<IntTy, Error> {
    let from_val = |ty: ValType| match ty {
        ValType::I32 => Ok(IntTy::I32),
        ValType::I64 => Ok(IntTy::I64),
        other => unsup(format!("non-integer operand type {}", other.as_str())),
    };
    match op {
        Operand::LocalOperand { ty, .. } => from_val(val_type(ty.as_ref())?),
        Operand::ConstantOperand(c) => match c.as_ref() {
            Constant::Int { bits, .. } if *bits <= 32 => Ok(IntTy::I32),
            Constant::Int { bits, .. } if *bits == 64 => Ok(IntTy::I64),
            other => unsup(format!("constant operand type {other:?}")),
        },
        Operand::MetadataOperand => unsup("metadata operand type"),
    }
}

fn translate_func(f: &llvm_ir::Function) -> Result<Func, Error> {
    if f.is_var_arg {
        return unsup(format!("varargs function `{}`", f.name));
    }
    let params: Vec<ValType> = f
        .parameters
        .iter()
        .map(|p| val_type(&p.ty))
        .collect::<Result<_, _>>()?;
    let results = match f.return_type.as_ref() {
        Type::VoidType => Vec::new(),
        t => vec![val_type(t)?],
    };
    // Milestone 0: single basic block only (φ → block params is Milestone 1).
    if f.basic_blocks.len() != 1 {
        return unsup(format!(
            "function `{}` has {} basic blocks (multi-block / control flow is Milestone 1)",
            f.name,
            f.basic_blocks.len()
        ));
    }
    let bb = &f.basic_blocks[0];

    // SSA value numbering is block-local (§3a): params occupy `0..params.len()`, then each
    // value-producing instruction (incl. materialized constants) takes the next index.
    let mut map: HashMap<Name, ValIdx> = HashMap::new();
    for (i, p) in f.parameters.iter().enumerate() {
        map.insert(p.name.clone(), i as ValIdx);
    }
    let mut ctx = BlockCtx {
        insts: Vec::new(),
        map,
        next_val: params.len() as ValIdx,
    };

    for instr in &bb.instrs {
        ctx.translate_inst(instr)?;
    }
    let term = ctx.translate_term(&bb.term)?;

    Ok(Func {
        params: params.clone(),
        results,
        // The entry block's params must equal the function signature's params (§3b).
        blocks: vec![Block {
            params,
            insts: ctx.insts,
            term,
        }],
    })
}

/// A block under construction: the straight-line body, the LLVM-name → SSA-index map, and the
/// running block-local value counter.
struct BlockCtx {
    insts: Vec<Inst>,
    map: HashMap<Name, ValIdx>,
    next_val: ValIdx,
}

impl BlockCtx {
    /// Append a value-producing instruction and return its block-local index.
    fn push(&mut self, inst: Inst) -> ValIdx {
        self.insts.push(inst);
        let idx = self.next_val;
        self.next_val += 1;
        idx
    }

    /// Resolve an operand to an SSA value index, materializing a constant as a `const` instruction
    /// (SVM has no constant pool — constants are instructions, §3b).
    fn operand(&mut self, op: &Operand) -> Result<ValIdx, Error> {
        match op {
            Operand::LocalOperand { name, .. } => self
                .map
                .get(name)
                .copied()
                .ok_or_else(|| Error::Unsupported(format!("unresolved local {name:?}"))),
            Operand::ConstantOperand(c) => match c.as_ref() {
                Constant::Int { bits, value } if *bits <= 32 => {
                    Ok(self.push(Inst::ConstI32(*value as u32 as i32)))
                }
                Constant::Int { bits, value } if *bits == 64 => {
                    Ok(self.push(Inst::ConstI64(*value as i64)))
                }
                other => unsup(format!("constant operand {other:?}")),
            },
            Operand::MetadataOperand => unsup("metadata operand"),
        }
    }

    fn translate_inst(&mut self, instr: &Instruction) -> Result<(), Error> {
        // Milestone 0: integer binary arithmetic/bitwise only. `nsw`/`nuw`/`exact` flags are
        // ignored — SVM defines wrapping semantics (§3b), and the flags only license LLVM UB we
        // do not reproduce (the §3c totality discipline).
        use Instruction as I;
        let (op0, op1, dest, binop) = match instr {
            I::Add(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::Add),
            I::Sub(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::Sub),
            I::Mul(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::Mul),
            I::And(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::And),
            I::Or(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::Or),
            I::Xor(x) => (&x.operand0, &x.operand1, &x.dest, BinOp::Xor),
            other => return unsup(format!("instruction {other:?}")),
        };
        let ty = operand_int_ty(op0)?;
        let a = self.operand(op0)?;
        let b = self.operand(op1)?;
        let idx = self.push(Inst::IntBin {
            ty,
            op: binop,
            a,
            b,
        });
        self.map.insert(dest.clone(), idx);
        Ok(())
    }

    fn translate_term(&mut self, term: &LTerm) -> Result<Terminator, Error> {
        match term {
            LTerm::Ret(r) => match &r.return_operand {
                None => Ok(Terminator::Return(Vec::new())),
                Some(op) => {
                    let v = self.operand(op)?;
                    Ok(Terminator::Return(vec![v]))
                }
            },
            // `unreachable` after UB → a defined trap to the host (§3b/§3c totality).
            LTerm::Unreachable(_) => Ok(Terminator::Unreachable),
            other => unsup(format!("terminator {other:?}")),
        }
    }
}
