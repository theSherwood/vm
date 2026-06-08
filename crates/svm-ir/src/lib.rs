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
    Rotl,
    Rotr,
}

impl BinOp {
    pub const ALL: [BinOp; 15] = [
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
        BinOp::Rotl,
        BinOp::Rotr,
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
            BinOp::Rotl => "rotl",
            BinOp::Rotr => "rotr",
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

/// Unary integer ops (same type in and out). `clz`/`ctz`/`popcnt` are bit counts;
/// `extendN_s` sign-extends the low N bits. (`extend32_s` on `i32` is the identity.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntUnOp {
    Clz,
    Ctz,
    Popcnt,
    Extend8S,
    Extend16S,
    Extend32S,
}

impl IntUnOp {
    pub const ALL: [IntUnOp; 6] = [
        IntUnOp::Clz,
        IntUnOp::Ctz,
        IntUnOp::Popcnt,
        IntUnOp::Extend8S,
        IntUnOp::Extend16S,
        IntUnOp::Extend32S,
    ];
    pub fn name(self) -> &'static str {
        match self {
            IntUnOp::Clz => "clz",
            IntUnOp::Ctz => "ctz",
            IntUnOp::Popcnt => "popcnt",
            IntUnOp::Extend8S => "extend8_s",
            IntUnOp::Extend16S => "extend16_s",
            IntUnOp::Extend32S => "extend32_s",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<IntUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<IntUnOp> {
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

/// Float width. Maps to the `f32`/`f64` text prefix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn val(self) -> ValType {
        match self {
            FloatTy::F32 => ValType::F32,
            FloatTy::F64 => ValType::F64,
        }
    }
    pub fn prefix(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }
}

/// Binary float ops (IEEE 754, no traps; same type in and out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Copysign,
}

impl FBinOp {
    pub const ALL: [FBinOp; 7] = [
        FBinOp::Add,
        FBinOp::Sub,
        FBinOp::Mul,
        FBinOp::Div,
        FBinOp::Min,
        FBinOp::Max,
        FBinOp::Copysign,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FBinOp::Add => "add",
            FBinOp::Sub => "sub",
            FBinOp::Mul => "mul",
            FBinOp::Div => "div",
            FBinOp::Min => "min",
            FBinOp::Max => "max",
            FBinOp::Copysign => "copysign",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Unary float ops (IEEE 754, no traps).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FUnOp {
    Abs,
    Neg,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

impl FUnOp {
    pub const ALL: [FUnOp; 7] = [
        FUnOp::Abs,
        FUnOp::Neg,
        FUnOp::Sqrt,
        FUnOp::Ceil,
        FUnOp::Floor,
        FUnOp::Trunc,
        FUnOp::Nearest,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FUnOp::Abs => "abs",
            FUnOp::Neg => "neg",
            FUnOp::Sqrt => "sqrt",
            FUnOp::Ceil => "ceil",
            FUnOp::Floor => "floor",
            FUnOp::Trunc => "trunc",
            FUnOp::Nearest => "nearest",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Float comparisons (same type in, `i32` 0/1 out).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl FCmpOp {
    pub const ALL: [FCmpOp; 6] = [
        FCmpOp::Eq,
        FCmpOp::Ne,
        FCmpOp::Lt,
        FCmpOp::Le,
        FCmpOp::Gt,
        FCmpOp::Ge,
    ];
    pub fn name(self) -> &'static str {
        match self {
            FCmpOp::Eq => "eq",
            FCmpOp::Ne => "ne",
            FCmpOp::Lt => "lt",
            FCmpOp::Le => "le",
            FCmpOp::Gt => "gt",
            FCmpOp::Ge => "ge",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FCmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FCmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Saturating float→int conversions (`trunc_sat`): NaN→0, out-of-range saturates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FToI {
    F32I32S,
    F32I32U,
    F32I64S,
    F32I64U,
    F64I32S,
    F64I32U,
    F64I64S,
    F64I64U,
}

impl FToI {
    pub const ALL: [FToI; 8] = [
        FToI::F32I32S,
        FToI::F32I32U,
        FToI::F32I64S,
        FToI::F32I64U,
        FToI::F64I32S,
        FToI::F64I32U,
        FToI::F64I64S,
        FToI::F64I64U,
    ];
    /// `(from float, to int, signed)`.
    pub fn parts(self) -> (FloatTy, IntTy, bool) {
        match self {
            FToI::F32I32S => (FloatTy::F32, IntTy::I32, true),
            FToI::F32I32U => (FloatTy::F32, IntTy::I32, false),
            FToI::F32I64S => (FloatTy::F32, IntTy::I64, true),
            FToI::F32I64U => (FloatTy::F32, IntTy::I64, false),
            FToI::F64I32S => (FloatTy::F64, IntTy::I32, true),
            FToI::F64I32U => (FloatTy::F64, IntTy::I32, false),
            FToI::F64I64S => (FloatTy::F64, IntTy::I64, true),
            FToI::F64I64U => (FloatTy::F64, IntTy::I64, false),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            FToI::F32I32S => "i32.trunc_sat_f32_s",
            FToI::F32I32U => "i32.trunc_sat_f32_u",
            FToI::F32I64S => "i64.trunc_sat_f32_s",
            FToI::F32I64U => "i64.trunc_sat_f32_u",
            FToI::F64I32S => "i32.trunc_sat_f64_s",
            FToI::F64I32U => "i32.trunc_sat_f64_u",
            FToI::F64I64S => "i64.trunc_sat_f64_s",
            FToI::F64I64U => "i64.trunc_sat_f64_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<FToI> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<FToI> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
    /// The **trapping** spelling (`trunc`, no `_sat`) of the same conversion — NaN
    /// and out-of-range inputs trap instead of saturating.
    pub fn trap_name(self) -> &'static str {
        match self {
            FToI::F32I32S => "i32.trunc_f32_s",
            FToI::F32I32U => "i32.trunc_f32_u",
            FToI::F32I64S => "i64.trunc_f32_s",
            FToI::F32I64U => "i64.trunc_f32_u",
            FToI::F64I32S => "i32.trunc_f64_s",
            FToI::F64I32U => "i32.trunc_f64_u",
            FToI::F64I64S => "i64.trunc_f64_s",
            FToI::F64I64U => "i64.trunc_f64_u",
        }
    }
    pub fn from_trap_name(s: &str) -> Option<FToI> {
        Self::ALL.iter().copied().find(|o| o.trap_name() == s)
    }
}

/// Int→float conversions (`convert`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IToF {
    I32F32S,
    I32F32U,
    I64F32S,
    I64F32U,
    I32F64S,
    I32F64U,
    I64F64S,
    I64F64U,
}

impl IToF {
    pub const ALL: [IToF; 8] = [
        IToF::I32F32S,
        IToF::I32F32U,
        IToF::I64F32S,
        IToF::I64F32U,
        IToF::I32F64S,
        IToF::I32F64U,
        IToF::I64F64S,
        IToF::I64F64U,
    ];
    /// `(from int, to float, signed)`.
    pub fn parts(self) -> (IntTy, FloatTy, bool) {
        match self {
            IToF::I32F32S => (IntTy::I32, FloatTy::F32, true),
            IToF::I32F32U => (IntTy::I32, FloatTy::F32, false),
            IToF::I64F32S => (IntTy::I64, FloatTy::F32, true),
            IToF::I64F32U => (IntTy::I64, FloatTy::F32, false),
            IToF::I32F64S => (IntTy::I32, FloatTy::F64, true),
            IToF::I32F64U => (IntTy::I32, FloatTy::F64, false),
            IToF::I64F64S => (IntTy::I64, FloatTy::F64, true),
            IToF::I64F64U => (IntTy::I64, FloatTy::F64, false),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            IToF::I32F32S => "f32.convert_i32_s",
            IToF::I32F32U => "f32.convert_i32_u",
            IToF::I64F32S => "f32.convert_i64_s",
            IToF::I64F32U => "f32.convert_i64_u",
            IToF::I32F64S => "f64.convert_i32_s",
            IToF::I32F64U => "f64.convert_i32_u",
            IToF::I64F64S => "f64.convert_i64_s",
            IToF::I64F64U => "f64.convert_i64_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<IToF> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<IToF> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Float width-change (`demote`/`promote`) and bit-`reinterpret` casts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CastOp {
    Demote,  // f64 -> f32
    Promote, // f32 -> f64
    ReinterpI32F32,
    ReinterpF32I32,
    ReinterpI64F64,
    ReinterpF64I64,
}

impl CastOp {
    pub const ALL: [CastOp; 6] = [
        CastOp::Demote,
        CastOp::Promote,
        CastOp::ReinterpI32F32,
        CastOp::ReinterpF32I32,
        CastOp::ReinterpI64F64,
        CastOp::ReinterpF64I64,
    ];
    /// `(text name, source type, result type)`.
    pub fn sig(self) -> (&'static str, ValType, ValType) {
        match self {
            CastOp::Demote => ("f32.demote_f64", ValType::F64, ValType::F32),
            CastOp::Promote => ("f64.promote_f32", ValType::F32, ValType::F64),
            CastOp::ReinterpI32F32 => ("f32.reinterpret_i32", ValType::I32, ValType::F32),
            CastOp::ReinterpF32I32 => ("i32.reinterpret_f32", ValType::F32, ValType::I32),
            CastOp::ReinterpI64F64 => ("f64.reinterpret_i64", ValType::I64, ValType::F64),
            CastOp::ReinterpF64I64 => ("i64.reinterpret_f64", ValType::F64, ValType::I64),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<CastOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<CastOp> {
        Self::ALL.iter().copied().find(|o| o.sig().0 == s)
    }
}

/// Memory load ops. Each reads `width` little-endian bytes at the confined effective
/// address and produces `result`; narrow integer loads sign- or zero-extend per
/// `signed` into the (i32/i64) result type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadOp {
    I32,
    I64,
    F32,
    F64,
    I32_8S,
    I32_8U,
    I32_16S,
    I32_16U,
    I64_8S,
    I64_8U,
    I64_16S,
    I64_16U,
    I64_32S,
    I64_32U,
}

impl LoadOp {
    pub const ALL: [LoadOp; 14] = [
        LoadOp::I32,
        LoadOp::I64,
        LoadOp::F32,
        LoadOp::F64,
        LoadOp::I32_8S,
        LoadOp::I32_8U,
        LoadOp::I32_16S,
        LoadOp::I32_16U,
        LoadOp::I64_8S,
        LoadOp::I64_8U,
        LoadOp::I64_16S,
        LoadOp::I64_16U,
        LoadOp::I64_32S,
        LoadOp::I64_32U,
    ];
    /// `(text name, result type, access width in bytes, sign-extended)`.
    pub fn info(self) -> (&'static str, ValType, u32, bool) {
        match self {
            LoadOp::I32 => ("i32.load", ValType::I32, 4, false),
            LoadOp::I64 => ("i64.load", ValType::I64, 8, false),
            LoadOp::F32 => ("f32.load", ValType::F32, 4, false),
            LoadOp::F64 => ("f64.load", ValType::F64, 8, false),
            LoadOp::I32_8S => ("i32.load8_s", ValType::I32, 1, true),
            LoadOp::I32_8U => ("i32.load8_u", ValType::I32, 1, false),
            LoadOp::I32_16S => ("i32.load16_s", ValType::I32, 2, true),
            LoadOp::I32_16U => ("i32.load16_u", ValType::I32, 2, false),
            LoadOp::I64_8S => ("i64.load8_s", ValType::I64, 1, true),
            LoadOp::I64_8U => ("i64.load8_u", ValType::I64, 1, false),
            LoadOp::I64_16S => ("i64.load16_s", ValType::I64, 2, true),
            LoadOp::I64_16U => ("i64.load16_u", ValType::I64, 2, false),
            LoadOp::I64_32S => ("i64.load32_s", ValType::I64, 4, true),
            LoadOp::I64_32U => ("i64.load32_u", ValType::I64, 4, false),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<LoadOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<LoadOp> {
        Self::ALL.iter().copied().find(|o| o.info().0 == s)
    }
}

/// Memory store ops. Each writes the low `width` little-endian bytes of the value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreOp {
    I32,
    I64,
    F32,
    F64,
    I32_8,
    I32_16,
    I64_8,
    I64_16,
    I64_32,
}

impl StoreOp {
    pub const ALL: [StoreOp; 9] = [
        StoreOp::I32,
        StoreOp::I64,
        StoreOp::F32,
        StoreOp::F64,
        StoreOp::I32_8,
        StoreOp::I32_16,
        StoreOp::I64_8,
        StoreOp::I64_16,
        StoreOp::I64_32,
    ];
    /// `(text name, value type, access width in bytes)`.
    pub fn info(self) -> (&'static str, ValType, u32) {
        match self {
            StoreOp::I32 => ("i32.store", ValType::I32, 4),
            StoreOp::I64 => ("i64.store", ValType::I64, 8),
            StoreOp::F32 => ("f32.store", ValType::F32, 4),
            StoreOp::F64 => ("f64.store", ValType::F64, 8),
            StoreOp::I32_8 => ("i32.store8", ValType::I32, 1),
            StoreOp::I32_16 => ("i32.store16", ValType::I32, 2),
            StoreOp::I64_8 => ("i64.store8", ValType::I64, 1),
            StoreOp::I64_16 => ("i64.store16", ValType::I64, 2),
            StoreOp::I64_32 => ("i64.store32", ValType::I64, 4),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<StoreOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<StoreOp> {
        Self::ALL.iter().copied().find(|o| o.info().0 == s)
    }
}

/// §12 atomic read-modify-write operation. Each atomically loads the operand, applies the op with
/// the argument, stores the result, and yields the **old** value (`Xchg` just swaps the argument in).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomicRmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

impl AtomicRmwOp {
    pub const ALL: [AtomicRmwOp; 6] = [
        AtomicRmwOp::Add,
        AtomicRmwOp::Sub,
        AtomicRmwOp::And,
        AtomicRmwOp::Or,
        AtomicRmwOp::Xor,
        AtomicRmwOp::Xchg,
    ];
    /// The text suffix in `<ty>.atomic.rmw.<suffix>`.
    pub fn name(self) -> &'static str {
        match self {
            AtomicRmwOp::Add => "add",
            AtomicRmwOp::Sub => "sub",
            AtomicRmwOp::And => "and",
            AtomicRmwOp::Or => "or",
            AtomicRmwOp::Xor => "xor",
            AtomicRmwOp::Xchg => "xchg",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<AtomicRmwOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<AtomicRmwOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// C11/§12 memory ordering for atomic ops and fences. The IR carries the full lattice so a frontend
/// can express it and the verifier can reject impossible op/ordering pairs; **both backends currently
/// execute every atomic sequentially-consistent** (a sound strengthening — Cranelift atomics are
/// seq-cst only, and it keeps the interpreter↔JIT oracle exact). Honoring weaker orderings in
/// execution awaits a backend that can, and the concurrent-oracle story (§18).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl Ordering {
    pub const ALL: [Ordering; 5] = [
        Ordering::Relaxed,
        Ordering::Acquire,
        Ordering::Release,
        Ordering::AcqRel,
        Ordering::SeqCst,
    ];
    /// The text suffix; the default [`Ordering::SeqCst`] is rendered by omitting the suffix entirely
    /// (so existing `.atomic.` text round-trips unchanged).
    pub fn name(self) -> &'static str {
        match self {
            Ordering::Relaxed => "relaxed",
            Ordering::Acquire => "acquire",
            Ordering::Release => "release",
            Ordering::AcqRel => "acqrel",
            Ordering::SeqCst => "seqcst",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<Ordering> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<Ordering> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
    /// A load may not carry release semantics (`Release`/`AcqRel`).
    pub fn valid_for_load(self) -> bool {
        !matches!(self, Ordering::Release | Ordering::AcqRel)
    }
    /// A store may not carry acquire semantics (`Acquire`/`AcqRel`).
    pub fn valid_for_store(self) -> bool {
        !matches!(self, Ordering::Acquire | Ordering::AcqRel)
    }
}

/// Non-terminator instructions. Each produces exactly one result — appended at the
/// next block-local value index — **except `Store`, which produces no value** (see
/// [`Inst::produces_value`]).
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
    /// Unary integer op; operand and result are `ty`.
    IntUn {
        ty: IntTy,
        op: IntUnOp,
        a: ValIdx,
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
    /// `f32`/`f64` constants, stored as raw bits for exact (NaN-safe) round-tripping.
    ConstF32(u32),
    ConstF64(u64),
    /// Binary float op; operands and result are `ty`.
    FBin {
        ty: FloatTy,
        op: FBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Unary float op; operand and result are `ty`.
    FUn {
        ty: FloatTy,
        op: FUnOp,
        a: ValIdx,
    },
    /// Float compare; operands are `ty`, result is `i32` 0/1.
    FCmp {
        ty: FloatTy,
        op: FCmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Saturating float→int conversion.
    FToISat {
        op: FToI,
        a: ValIdx,
    },
    /// Trapping float→int conversion: NaN or out-of-range input traps (vs the
    /// saturating [`Inst::FToISat`] default).
    FToITrap {
        op: FToI,
        a: ValIdx,
    },
    /// Int→float conversion.
    IToFConv {
        op: IToF,
        a: ValIdx,
    },
    /// `demote`/`promote`/`reinterpret` cast.
    Cast {
        op: CastOp,
        a: ValIdx,
    },
    /// Load `op`'s width from the confined effective address `addr + offset`.
    /// `align` is a power-of-two alignment *hint* (log2); it does not affect
    /// semantics (unaligned access is allowed). Confinement masking is implicit.
    Load {
        op: LoadOp,
        addr: ValIdx,
        offset: u64,
        align: u8,
    },
    /// Store `value` (`op`'s width) at the confined effective address. Produces no
    /// SSA result. `align` is a hint (see [`Inst::Load`]).
    Store {
        op: StoreOp,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
        align: u8,
    },
    /// §12 atomic load — a naturally-aligned read of `ty` from the confined effective address
    /// `addr + offset`; a misaligned effective address **traps**. Single-threaded value semantics
    /// equal a plain [`Inst::Load`]; the distinct op makes the JIT emit a hardware atomic
    /// (sequentially consistent), so it stays correct once threads exist (§12).
    AtomicLoad {
        ty: IntTy,
        addr: ValIdx,
        offset: u64,
        order: Ordering,
    },
    /// §12 atomic store — a naturally-aligned write of `value` (`ty`) to `addr + offset`; a
    /// misaligned effective address **traps**. Produces no SSA result (like [`Inst::Store`]).
    AtomicStore {
        ty: IntTy,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
        order: Ordering,
    },
    /// §12 atomic read-modify-write: atomically apply `op` with `value` to `*(addr+offset)`
    /// (`ty`-wide, naturally aligned ⇒ else **traps**) and yield the **old** value.
    AtomicRmw {
        ty: IntTy,
        op: AtomicRmwOp,
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
        order: Ordering,
    },
    /// §12 atomic compare-exchange: if `*(addr+offset) == expected`, store `replacement`; always
    /// yield the **old** value (`ty`-wide, naturally aligned ⇒ else **traps**).
    AtomicCmpxchg {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        replacement: ValIdx,
        offset: u64,
        order: Ordering,
    },
    /// Direct call to a function by index (fully static; the verifier checks the
    /// index and argument types). Appends the callee's result values — **0, 1, or
    /// many** — at the next block-local indices.
    Call {
        func: FuncIdx,
        args: Vec<ValIdx>,
    },
    /// `ref.func`: materialize a function reference — just the function index as an
    /// `i32` (a `funcref` is a forgeable integer, §3c). The verifier checks the index
    /// is in range; the *value* is plain data.
    RefFunc {
        func: FuncIdx,
    },
    /// Indirect call through the function table (§3c): mask `idx` into the table,
    /// runtime-check the selected function's signature against `ty`, then call.
    /// `idx` is an `i32` table index; results are `ty.results`.
    CallIndirect {
        ty: FuncType,
        idx: ValIdx,
        args: Vec<ValIdx>,
    },
    /// Pointer arithmetic: `ptr + integer_offset`. Off-CHERI a plain `i64` wrapping
    /// add; the distinct opcode lets the JIT/CHERI backend see pointer provenance
    /// (§3b/§10). Operands and result are `i64`.
    PtrAdd {
        a: ValIdx,
        b: ValIdx,
    },
    /// `ptr.from_int` (`to_int = false`) / `ptr.to_int` (`to_int = true`): a free,
    /// no-op `i64`↔`i64` provenance cast off-CHERI (§3a/§10).
    PtrCast {
        to_int: bool,
        a: ValIdx,
    },
    /// Capability call (§3c): invoke operation `op` of the interface identified by
    /// `type_id` on the capability named by `handle` — a forgeable `i32` index into
    /// the **host-owned** handle table. At this use site the index is masked into the
    /// table and the entry's `type_id`/generation are re-checked, so a forged index is
    /// **inert**: it traps (wrong type / dead generation) or selects one of this
    /// domain's own granted `type_id` capabilities — never host memory or arbitrary
    /// code (§3c). `sig` is the operation's static signature; its results are appended.
    ///
    /// Phase-1 simplification: `type_id`/`op`/`sig` are inlined immediates (mirroring
    /// `call_indirect`'s inlined `FuncType`). A module-level interface/type section —
    /// which would let the verifier also bound `op` and cross-check `sig` against the
    /// canonical interface — is deferred to §13 linking. Safety does **not** depend on
    /// it: the host-owned table's use-site checks carry it, and the host handler
    /// treats all guest inputs as hostile (§2a authority-TCB).
    CapCall {
        type_id: u32,
        op: u32,
        sig: FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// §12 fiber create (`cont.new`): allocate a new suspended fiber that will run the
    /// function referenced by `func` on the data stack based at `sp`. `func` is an `i32`
    /// funcref, resolved through the function table with signature `(i64 sp, i64 arg) ->
    /// i64` at first resume (a bad ref traps there, like [`Inst::CallIndirect`]); `sp`
    /// (`i64`) is the fiber's own data-stack base — a fiber owns a **stack pair** (§3d): its
    /// in-window data stack (based here) plus the out-of-band control stack the runtime
    /// allocates. Yields an `i32` **fiber handle**: a forgeable index into the runtime-owned
    /// fiber table, masked + generation-checked at use like a capability handle (§3c), so a
    /// forged handle is inert (it traps or selects one of this domain's own fibers, never
    /// host state). The fiber does not run yet; the first resume calls `func(sp, arg)`.
    ContNew {
        func: ValIdx,
        sp: ValIdx,
    },
    /// §12 fiber resume (`cont.resume`): switch to fiber `k` (an `i32` handle), delivering
    /// `arg` (`i64`) — the argument to the fiber's function on the first resume, or the
    /// result of the fiber's `suspend` on later resumes. Runs the fiber until it suspends
    /// or returns, then yields `(status: i32, value: i64)`: `status` 0 = **suspended** (the
    /// fiber stays resumable), 1 = **returned** (the fiber is done; resuming it again
    /// traps). A **call-clobbering** control op — like a call it switches stacks, but it
    /// does not end the block.
    ContResume {
        k: ValIdx,
        arg: ValIdx,
    },
    /// §12 fiber suspend (`suspend`): from within a running fiber, suspend back to the
    /// resumer delivering `value` (`i64`); evaluates to the `i64` `arg` of the next resume.
    /// Suspending when no fiber is running (the root computation) **traps**. Like
    /// [`Inst::ContResume`] this is a call-clobbering control op.
    Suspend {
        value: ValIdx,
    },
    /// §12 thread spawn (`thread.spawn`): start a new vCPU — **one real OS thread** (1:1; the VM
    /// provides the thread + futex as *primitives*, not a scheduler — any M:N model is built by the
    /// guest runtime over `thread.spawn` + `cont.*`, D22) — running `funcs[func]` on the data stack
    /// based at `sp` (the §3d two-stack split — every vCPU owns its own in-window data stack, exactly
    /// like a fiber) with `arg`, over the **same** guest memory (anonymous `Region` bytes and §13
    /// aliases are shared; post-spawn mapping changes are thread-local for now). `func` must have the
    /// fixed thread-entry type `(i64 sp, i64 arg) -> i64` (verifier-checked) — the same signature as a
    /// fiber, so a frontend function works as-is.
    /// Yields an `i32` **thread handle**: a forgeable index into the runtime-owned thread table,
    /// masked + generation-checked at [`Inst::ThreadJoin`] like a fiber/capability handle (§3c), so a
    /// forged handle is inert (it traps).
    ThreadSpawn {
        func: FuncIdx,
        sp: ValIdx,
        arg: ValIdx,
    },
    /// §12 thread join (`thread.join`): block until the vCPU named by `handle` (an `i32` thread
    /// handle) finishes and yield its `i64` result. A forged / out-of-range / already-joined handle
    /// is inert (**traps**); if the joined vCPU itself trapped, that trap propagates here.
    ThreadJoin {
        handle: ValIdx,
    },
    /// §12 futex wait (`<ty>.atomic.wait`): if the `ty`-wide value at the confined, naturally-aligned
    /// address `addr` still equals `expected`, block this vCPU until a [`Inst::MemoryNotify`] on the
    /// same address wakes it or `timeout` nanoseconds (`i64`) elapse. Yields an `i32` status: `0` =
    /// woken by a notify, `1` = the value did not equal `expected` (no wait), `2` = timed out. A
    /// misaligned address **traps** (like the other atomics).
    MemoryWait {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        timeout: ValIdx,
    },
    /// §12 futex notify (`atomic.notify`): wake up to `count` (`i32`) vCPUs waiting on the confined
    /// address `addr`. Yields an `i32`: the number of waiters woken (capped at `count`). Accesses no
    /// memory, so it never faults on protection — only the address is confined.
    MemoryNotify {
        addr: ValIdx,
        count: ValIdx,
    },
    /// §12 standalone memory fence (`atomic.fence <order>`): orders this vCPU's accesses without
    /// touching memory. Produces no SSA result. Honored by the interpreter; the JIT does not yet
    /// lower it (interp-only, like fibers).
    AtomicFence {
        order: Ordering,
    },
}

impl Inst {
    /// How many values this instruction appends at the next block-local indices.
    ///
    /// Most instructions append exactly one; `Store` appends none; a `Call` appends
    /// its callee's result count, so it needs the per-function result arities
    /// (indexed by [`FuncIdx`]) to answer; `CallIndirect` carries its own signature.
    pub fn result_count(&self, fn_results: &[usize]) -> usize {
        match self {
            Inst::Store { .. } | Inst::AtomicStore { .. } | Inst::AtomicFence { .. } => 0,
            // `cont.resume` is the one multi-result non-call op: `(status, value)`.
            Inst::ContResume { .. } => 2,
            Inst::Call { func, .. } => fn_results.get(*func as usize).copied().unwrap_or(0),
            Inst::CallIndirect { ty, .. } => ty.results.len(),
            Inst::CapCall { sig, .. } => sig.results.len(),
            _ => 1,
        }
    }
}

/// A function signature — the immediate carried by `call_indirect` and (later) the
/// function-table type ids. Equality is structural (the runtime "type_id" check).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
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
    /// Tail call (`return_call`): replace the current frame with a direct callee
    /// whose results are this function's results. Args match the callee's params.
    ReturnCall { func: FuncIdx, args: Vec<ValIdx> },
    /// Indirect tail call (`return_call_indirect`): like [`Terminator::ReturnCall`]
    /// but dispatched through the function table (masked + signature-checked, §3c).
    ReturnCallIndirect {
        ty: FuncType,
        idx: ValIdx,
        args: Vec<ValIdx>,
    },
    /// Abort: control must not reach here. Delivers a trap to the host (§3b/§5).
    /// Covers both `unreachable` and language-level `trap`/`assert` failure.
    Unreachable,
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

/// A linear-memory window declaration (§4). The window is `1 << size_log2` bytes —
/// a power of two, so confinement is a single `addr & (size − 1)` mask. The window
/// is a reserved virtual range; guest pointers are offsets into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Memory {
    pub size_log2: u8,
}

impl Memory {
    /// Window size in bytes (`1 << size_log2`). `size_log2` is verified `< 64`.
    pub fn size(self) -> u64 {
        1u64 << self.size_log2
    }
    /// The confinement mask (`size − 1`).
    pub fn mask(self) -> u64 {
        self.size() - 1
    }
}

/// The reference host's **default reservation policy** (§4): the size (`log2`) of the reserved
/// virtual range a window's `mapped` bytes live inside. DESIGN §4 makes this host-configurable
/// ("e.g. 2^40"); this is the default the reference `run`/`compile_and_run` entries apply when a
/// caller doesn't pass one. It is *policy*, not verified semantics — both backends share this one
/// constant so they stay in differential lockstep, and the masking unit (`svm-mask`) never
/// hard-codes a reservation. The reserved range is `PROT_NONE` + lazily paged, so a large value
/// costs virtual address space, not committed memory.
pub const DEFAULT_RESERVED_LOG2: u8 = 40;

/// A module: a flat list of functions plus an optional linear-memory window.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Module {
    pub funcs: Vec<Func>,
    pub memory: Option<Memory>,
    /// Initialized data segments placed in the window at instantiation (§3a): each writes
    /// `bytes` at `offset`, and a `readonly` segment is then mapped read-only (D40 — a write
    /// to it faults, §4/§5). Like an ELF loader laying out `.data`/`.rodata`; replaces the
    /// frontend's per-byte `_start` init stores.
    pub data: Vec<Data>,
}

/// An initialized data segment (§3a / D40). Placed in the window `[offset, offset+bytes.len())`
/// at instantiation; `readonly` ones are protected after the copy so guest writes fault.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Data {
    pub offset: u64,
    pub readonly: bool,
    pub bytes: Vec<u8>,
}
