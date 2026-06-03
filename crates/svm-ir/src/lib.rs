//! Core IR: block-local typed SSA over a CFG of basic blocks.
//!
//! See `DESIGN.md` §3/§3a/§3b. Key disciplines encoded here:
//! - Values are **block-local**: within a block, indices run `0..k` for the block
//!   parameters, then one more per instruction result. Operands reference *earlier*
//!   same-block indices only. Cross-block dataflow is *only* via block parameters,
//!   so dominance analysis is impossible to need (verifier is a linear pass).
//! - Every block ends in exactly one terminator.
//!
//! Phase-1 integer core: `i32`/`i64` constants, the full integer arithmetic /
//! bitwise / shift / comparison set, `i32`↔`i64` conversions, `select`, and the
//! `br`/`br_if`/`br_table`/`return` terminators. Float, memory, calls, and
//! capabilities come in later batches per §3b.
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

/// The integer width an op operates at. Maps to the `i32`/`i64` text prefix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntTy {
    I32,
    I64,
}

impl IntTy {
    pub fn val(self) -> ValType {
        match self {
            IntTy::I32 => ValType::I32,
            IntTy::I64 => ValType::I64,
        }
    }
    pub fn prefix(self) -> &'static str {
        match self {
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
        }
    }
}

/// Binary integer ops (same type in, same type out). Wrapping arithmetic; `div`/`rem`
/// trap on `/0` and on `INT_MIN/-1` (signed); shifts take the amount mod bitwidth.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
}

impl BinOp {
    pub const ALL: [BinOp; 13] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::DivS,
        BinOp::DivU,
        BinOp::RemS,
        BinOp::RemU,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::ShrS,
        BinOp::ShrU,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::DivS => "div_s",
            BinOp::DivU => "div_u",
            BinOp::RemS => "rem_s",
            BinOp::RemU => "rem_u",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::ShrS => "shr_s",
            BinOp::ShrU => "shr_u",
        }
    }

    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<BinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<BinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Integer comparisons (same type in, `i32` 0/1 out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    LtS,
    LtU,
    LeS,
    LeU,
    GtS,
    GtU,
    GeS,
    GeU,
}

impl CmpOp {
    pub const ALL: [CmpOp; 10] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::LtS,
        CmpOp::LtU,
        CmpOp::LeS,
        CmpOp::LeU,
        CmpOp::GtS,
        CmpOp::GtU,
        CmpOp::GeS,
        CmpOp::GeU,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::LtS => "lt_s",
            CmpOp::LtU => "lt_u",
            CmpOp::LeS => "le_s",
            CmpOp::LeU => "le_u",
            CmpOp::GtS => "gt_s",
            CmpOp::GtU => "gt_u",
            CmpOp::GeS => "ge_s",
            CmpOp::GeU => "ge_u",
        }
    }

    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<CmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<CmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Width-changing integer conversions between `i32` and `i64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvOp {
    /// `i64.extend_i32_s`: sign-extend `i32` → `i64`.
    ExtendI32S,
    /// `i64.extend_i32_u`: zero-extend `i32` → `i64`.
    ExtendI32U,
    /// `i32.wrap_i64`: truncate `i64` → `i32`.
    WrapI64,
}

impl ConvOp {
    /// `(text name, source type, result type)`.
    pub fn sig(self) -> (&'static str, ValType, ValType) {
        match self {
            ConvOp::ExtendI32S => ("i64.extend_i32_s", ValType::I32, ValType::I64),
            ConvOp::ExtendI32U => ("i64.extend_i32_u", ValType::I32, ValType::I64),
            ConvOp::WrapI64 => ("i32.wrap_i64", ValType::I64, ValType::I32),
        }
    }
    pub fn from_name(s: &str) -> Option<ConvOp> {
        [ConvOp::ExtendI32S, ConvOp::ExtendI32U, ConvOp::WrapI64]
            .into_iter()
            .find(|o| o.sig().0 == s)
    }
}

/// Non-terminator instructions. Each produces exactly one result whose index is
/// the next block-local value index (implicit, by position).
#[derive(Clone, PartialEq, Debug)]
pub enum Inst {
    ConstI32(i32),
    ConstI64(i64),
    /// Binary integer op; operands and result are `ty`.
    IntBin {
        ty: IntTy,
        op: BinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Integer compare; operands are `ty`, result is `i32` 0/1.
    IntCmp {
        ty: IntTy,
        op: CmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// `T.eqz`: 1 if the operand is zero else 0; result `i32`.
    Eqz {
        ty: IntTy,
        a: ValIdx,
    },
    /// Width conversion (see `ConvOp`).
    Convert {
        op: ConvOp,
        a: ValIdx,
    },
    /// Branchless choice: `cond` is `i32`; `a`/`b` share a type `T`; result `T`.
    Select {
        cond: ValIdx,
        a: ValIdx,
        b: ValIdx,
    },
}

/// One branch edge: a target block plus the argument values for its parameters.
pub type Edge = (BlockIdx, Vec<ValIdx>);

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
    /// Indexed multi-way branch. `idx` (`i32`) selects `targets[idx]`, or `default`
    /// when out of range. Each edge carries its own block arguments.
    BrTable {
        idx: ValIdx,
        targets: Vec<Edge>,
        default: Edge,
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
