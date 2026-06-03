//! Core IR: block-local typed SSA over a CFG of basic blocks.
//!
//! See `DESIGN.md` §3/§3a/§3b. Key disciplines encoded here:
//! - Values are **block-local**: within a block, indices run `0..k` for the block
//!   parameters, then one more per instruction result. Operands reference *earlier*
//!   same-block indices only. Cross-block dataflow is *only* via block parameters,
//!   so dominance analysis is impossible to need (verifier is a linear pass).
//! - Every block ends in exactly one terminator.
//!
//! This is the Phase-1 *slice*: a deliberately small instruction set
//! (`i32`/`i64` const + add, `br`/`br_if`/`return`) chosen to close the
//! text -> binary -> verify -> interp loop end to end. Extend per §3b.
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

/// Block-local value index (parameters first, then instruction results in order).
pub type ValIdx = u32;
/// Index of a block within a function (`0` = entry).
pub type BlockIdx = u32;
/// Index of a function within a module.
pub type FuncIdx = u32;

/// SSA value types. `i8`/`i16` are memory access *widths*, not value types (§3a).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl ValType {
    /// Stable text token (the text form is 1:1 with the binary, §3a).
    pub fn as_str(self) -> &'static str {
        match self {
            ValType::I32 => "i32",
            ValType::I64 => "i64",
            ValType::F32 => "f32",
            ValType::F64 => "f64",
        }
    }

    /// Parse a type token, if recognized.
    #[allow(clippy::should_implement_trait)] // `Option` return, not `FromStr`'s `Result`
    pub fn from_str(s: &str) -> Option<ValType> {
        Some(match s {
            "i32" => ValType::I32,
            "i64" => ValType::I64,
            "f32" => ValType::F32,
            "f64" => ValType::F64,
            _ => return None,
        })
    }
}

/// Non-terminator instructions. Each produces exactly one result whose index is
/// the next block-local value index (implicit, by position).
#[derive(Clone, PartialEq, Debug)]
pub enum Inst {
    I32Const(i32),
    I64Const(i64),
    /// `i32.add a b` — both operands must be `i32`; result `i32`.
    I32Add(ValIdx, ValIdx),
    /// `i64.add a b` — both operands must be `i64`; result `i64`.
    I64Add(ValIdx, ValIdx),
}

impl Inst {
    /// The result type this opcode produces given well-typed operands (§3a:
    /// "inferred result types"). Operand-type *checking* is the verifier's job.
    pub fn result_type(&self) -> ValType {
        match self {
            Inst::I32Const(_) | Inst::I32Add(..) => ValType::I32,
            Inst::I64Const(_) | Inst::I64Add(..) => ValType::I64,
        }
    }
}

/// Block terminators. Exactly one per block; only at the block end.
#[derive(Clone, PartialEq, Debug)]
pub enum Terminator {
    /// Unconditional branch with block arguments.
    Br { target: BlockIdx, args: Vec<ValIdx> },
    /// Two-target conditional branch (no implicit fallthrough, §3b). `cond` is `i32`.
    BrIf {
        cond: ValIdx,
        then_blk: BlockIdx,
        then_args: Vec<ValIdx>,
        else_blk: BlockIdx,
        else_args: Vec<ValIdx>,
    },
    /// Return values matching the function's result signature.
    Return(Vec<ValIdx>),
}

/// A basic block: a typed parameter list, a straight-line body, one terminator.
#[derive(Clone, PartialEq, Debug)]
pub struct Block {
    pub params: Vec<ValType>,
    pub insts: Vec<Inst>,
    pub term: Terminator,
}

/// A function: signature plus its blocks (`blocks[0]` is the entry block, whose
/// parameter types must equal the function's parameter types — §3b).
#[derive(Clone, PartialEq, Debug)]
pub struct Func {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
    pub blocks: Vec<Block>,
}

/// A module: a flat list of functions (the Phase-1 slice has no other sections yet).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Module {
    pub funcs: Vec<Func>,
}
