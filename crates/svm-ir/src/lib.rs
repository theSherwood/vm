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

/// Reserved pseudo-`type_id` for §7 capability **reflection** (`cap.self.*`). It is not a real
/// capability (no handle ever carries it): both backends lower `cap.self.count`/`cap.self.get` to a
/// host `cap.call` with this `type_id` (op 0 = count, op 1 = get), which the host's dispatch services
/// directly — read-only over the calling domain's own table — instead of resolving a handle. Sharing
/// one host entry point keeps the interpreter and JIT in lockstep. (Equivalent to issuing the
/// intrinsic, since reflection is ambient/authority-neutral; `u32::MAX` collides with no interface.)
pub const CAP_SELF_TYPE_ID: u32 = u32::MAX;

/// SSA value types. `i8`/`i16` are memory access *widths*, not value types (§3a).
/// `v128` is the fixed-128 SIMD vector (§17/D58): a first-class value carrying 16
/// raw bytes whose lane interpretation is per-op, never per-value.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
    V128,
    /// An opaque 64-bit **reference** (§GC.md §6 forward-compat reservation). Reserved now so
    /// future *precise* GC (stack maps + value-location metadata) can name pointer-typed slots
    /// without a format break. Today it is a pure reservation: no instruction produces a `ref`
    /// literal, and wherever a `ref` value does flow (a `ref`-typed param/result/block-arg) it is
    /// indistinguishable from an `i64` — it lowers as `i64` in the JIT and as the opaque
    /// `Value::Ref` in the interp. Conservative GC needs none of this; it scans raw words.
    Ref,
}

impl ValType {
    /// Stable text token (the text form is 1:1 with the binary, §3a).
    pub fn as_str(self) -> &'static str {
        match self {
            ValType::I32 => "i32",
            ValType::I64 => "i64",
            ValType::F32 => "f32",
            ValType::F64 => "f64",
            ValType::V128 => "v128",
            ValType::Ref => "ref",
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
            "v128" => ValType::V128,
            "ref" => ValType::Ref,
            _ => return None,
        })
    }
}

/// A `v128` **lane shape** (§17/D58): how a 16-byte vector is split into typed lanes
/// for one op. The shape is carried by the op, never by the `v128` value itself — the
/// same bytes are reinterpreted per instruction, exactly like hardware SIMD.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VShape {
    I8x16,
    I16x8,
    I32x4,
    I64x2,
    F32x4,
    F64x2,
}

impl VShape {
    pub const ALL: [VShape; 6] = [
        VShape::I8x16,
        VShape::I16x8,
        VShape::I32x4,
        VShape::I64x2,
        VShape::F32x4,
        VShape::F64x2,
    ];
    /// Number of lanes.
    pub fn lanes(self) -> u8 {
        match self {
            VShape::I8x16 => 16,
            VShape::I16x8 => 8,
            VShape::I32x4 | VShape::F32x4 => 4,
            VShape::I64x2 | VShape::F64x2 => 2,
        }
    }
    /// Lane width in bytes.
    pub fn lane_bytes(self) -> u32 {
        match self {
            VShape::I8x16 => 1,
            VShape::I16x8 => 2,
            VShape::I32x4 | VShape::F32x4 => 4,
            VShape::I64x2 | VShape::F64x2 => 8,
        }
    }
    /// Whether the lanes are floating-point.
    pub fn is_float(self) -> bool {
        matches!(self, VShape::F32x4 | VShape::F64x2)
    }
    /// The **scalar** value type a lane extracts to / splats from / replaces with.
    /// Narrow integer lanes (`i8`/`i16`) widen to `i32` (the lane scalar is an `i32`),
    /// matching the wasm/hardware convention.
    pub fn lane_val(self) -> ValType {
        match self {
            VShape::I8x16 | VShape::I16x8 | VShape::I32x4 => ValType::I32,
            VShape::I64x2 => ValType::I64,
            VShape::F32x4 => ValType::F32,
            VShape::F64x2 => ValType::F64,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            VShape::I8x16 => "i8x16",
            VShape::I16x8 => "i16x8",
            VShape::I32x4 => "i32x4",
            VShape::I64x2 => "i64x2",
            VShape::F32x4 => "f32x4",
            VShape::F64x2 => "f64x2",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VShape> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VShape> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }

    /// The integer shape with **half** the lane width (and twice the lanes): `i16x8`→`i8x16`,
    /// `i32x4`→`i16x8`, `i64x2`→`i32x4`. `None` for `i8x16` and the float shapes. The source of a
    /// widen / the result of a narrow.
    pub fn narrower(self) -> Option<VShape> {
        match self {
            VShape::I16x8 => Some(VShape::I8x16),
            VShape::I32x4 => Some(VShape::I16x8),
            VShape::I64x2 => Some(VShape::I32x4),
            _ => None,
        }
    }

    /// The integer shape with **double** the lane width: `i8x16`→`i16x8`, `i16x8`→`i32x4`,
    /// `i32x4`→`i64x2`. `None` for `i64x2` and the float shapes. The result of a widen / the source
    /// of a narrow.
    pub fn wider(self) -> Option<VShape> {
        match self {
            VShape::I8x16 => Some(VShape::I16x8),
            VShape::I16x8 => Some(VShape::I32x4),
            VShape::I32x4 => Some(VShape::I64x2),
            _ => None,
        }
    }
}

/// Lane-wise binary integer ops on a `v128` (§17). Defined for every integer [`VShape`]
/// (the JIT may lower a shape to several instructions — e.g. `i64x2.mul` — but the lane
/// semantics are always total). Wrapping arithmetic; shifts take the scalar amount mod
/// the lane bit-width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VIntBinOp {
    Add,
    Sub,
    Mul,
    MinS,
    MinU,
    MaxS,
    MaxU,
}

impl VIntBinOp {
    pub const ALL: [VIntBinOp; 7] = [
        VIntBinOp::Add,
        VIntBinOp::Sub,
        VIntBinOp::Mul,
        VIntBinOp::MinS,
        VIntBinOp::MinU,
        VIntBinOp::MaxS,
        VIntBinOp::MaxU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VIntBinOp::Add => "add",
            VIntBinOp::Sub => "sub",
            VIntBinOp::Mul => "mul",
            VIntBinOp::MinS => "min_s",
            VIntBinOp::MinU => "min_u",
            VIntBinOp::MaxS => "max_s",
            VIntBinOp::MaxU => "max_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VIntBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VIntBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise integer **comparison** ops on a `v128` (§17): each lane yields an all-ones (true) or
/// all-zeros (false) mask of the lane width, so the result is a `v128`. `s`/`u` select signed vs
/// unsigned lane ordering (`Eq`/`Ne` are sign-agnostic). Defined for every integer [`VShape`] — the
/// wasm spec omits unsigned `i64x2` compares, but the op set is total and the transpiler only emits
/// the shapes wasm defines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VICmpOp {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

impl VICmpOp {
    pub const ALL: [VICmpOp; 10] = [
        VICmpOp::Eq,
        VICmpOp::Ne,
        VICmpOp::LtS,
        VICmpOp::LtU,
        VICmpOp::GtS,
        VICmpOp::GtU,
        VICmpOp::LeS,
        VICmpOp::LeU,
        VICmpOp::GeS,
        VICmpOp::GeU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VICmpOp::Eq => "eq",
            VICmpOp::Ne => "ne",
            VICmpOp::LtS => "lt_s",
            VICmpOp::LtU => "lt_u",
            VICmpOp::GtS => "gt_s",
            VICmpOp::GtU => "gt_u",
            VICmpOp::LeS => "le_s",
            VICmpOp::LeU => "le_u",
            VICmpOp::GeS => "ge_s",
            VICmpOp::GeU => "ge_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VICmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VICmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **float** comparison ops on a `v128` (§17): each lane yields an all-ones (true) or
/// all-zeros (false) mask of the lane width → result `v128`. Defined for the float [`VShape`]s
/// (`f32x4`/`f64x2`). `eq`/`lt`/`gt`/`le`/`ge` are **ordered** (a NaN operand ⇒ false); `ne` is the
/// **unordered** negation (a NaN operand ⇒ true) — exactly the wasm (and Rust `==`/`!=`/`<`/…) rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFCmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl VFCmpOp {
    pub const ALL: [VFCmpOp; 6] = [
        VFCmpOp::Eq,
        VFCmpOp::Ne,
        VFCmpOp::Lt,
        VFCmpOp::Gt,
        VFCmpOp::Le,
        VFCmpOp::Ge,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VFCmpOp::Eq => "eq",
            VFCmpOp::Ne => "ne",
            VFCmpOp::Lt => "lt",
            VFCmpOp::Gt => "gt",
            VFCmpOp::Le => "le",
            VFCmpOp::Ge => "ge",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFCmpOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFCmpOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise integer **shift** ops on a `v128` (§17): every lane is shifted by the **same** scalar
/// `i32` amount, taken **modulo the lane bit-width** (the wasm rule). `ShrS` is arithmetic
/// (sign-replicating); `Shl`/`ShrU` are logical. Defined for every integer [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VShiftOp {
    Shl,
    ShrS,
    ShrU,
}

impl VShiftOp {
    pub const ALL: [VShiftOp; 3] = [VShiftOp::Shl, VShiftOp::ShrS, VShiftOp::ShrU];
    pub fn name(self) -> &'static str {
        match self {
            VShiftOp::Shl => "shl",
            VShiftOp::ShrS => "shr_s",
            VShiftOp::ShrU => "shr_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VShiftOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VShiftOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **unary** integer ops on a `v128` (§17): `Abs` (`|x|`, two's-complement, so
/// `abs(INT_MIN) == INT_MIN`, the wasm/hardware wrap) and `Neg` (`0 - x`, wrapping). `a`/result are
/// `v128`. Defined for every integer [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VIntUnOp {
    Abs,
    Neg,
}

impl VIntUnOp {
    pub const ALL: [VIntUnOp; 2] = [VIntUnOp::Abs, VIntUnOp::Neg];
    pub fn name(self) -> &'static str {
        match self {
            VIntUnOp::Abs => "abs",
            VIntUnOp::Neg => "neg",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VIntUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VIntUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **saturating** add/sub on a `v128` (§17): a lane that would overflow clamps to the
/// lane's signed/unsigned min or max instead of wrapping. Defined **only for `i8x16`/`i16x8`** (the
/// wasm spec has no wider saturating add/sub) — the verifier rejects any other [`VShape`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VSatBinOp {
    AddS,
    AddU,
    SubS,
    SubU,
}

impl VSatBinOp {
    pub const ALL: [VSatBinOp; 4] = [
        VSatBinOp::AddS,
        VSatBinOp::AddU,
        VSatBinOp::SubS,
        VSatBinOp::SubU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VSatBinOp::AddS => "add_sat_s",
            VSatBinOp::AddU => "add_sat_u",
            VSatBinOp::SubS => "sub_sat_s",
            VSatBinOp::SubU => "sub_sat_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VSatBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VSatBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **widening** (`extend`): take the low or high half of the source lanes and sign/zero-extend
/// each to twice the width. The result [`VShape`] is the wider one; the source is its [`VShape::narrower`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VWidenOp {
    LowS,
    LowU,
    HighS,
    HighU,
}

impl VWidenOp {
    pub const ALL: [VWidenOp; 4] = [
        VWidenOp::LowS,
        VWidenOp::LowU,
        VWidenOp::HighS,
        VWidenOp::HighU,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VWidenOp::LowS => "extend_low_s",
            VWidenOp::LowU => "extend_low_u",
            VWidenOp::HighS => "extend_high_s",
            VWidenOp::HighU => "extend_high_u",
        }
    }
    /// `(low_half, signed)`.
    pub fn parts(self) -> (bool, bool) {
        match self {
            VWidenOp::LowS => (true, true),
            VWidenOp::LowU => (true, false),
            VWidenOp::HighS => (false, true),
            VWidenOp::HighU => (false, false),
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VWidenOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VWidenOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **narrowing**: take two source vectors (each the wider shape), saturate every lane to the
/// narrow width, and concatenate (`a`'s lanes then `b`'s). `S`/`U` pick the *saturation* range; the
/// source is always read as **signed** (the wasm rule). `i8x16`/`i16x8` results only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VNarrowOp {
    S,
    U,
}

impl VNarrowOp {
    pub const ALL: [VNarrowOp; 2] = [VNarrowOp::S, VNarrowOp::U];
    pub fn name(self) -> &'static str {
        match self {
            VNarrowOp::S => "narrow_s",
            VNarrowOp::U => "narrow_u",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VNarrowOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VNarrowOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane **int↔float / float↔float conversions** (§17). Each is a whole-instruction mnemonic (the
/// source and result lane shapes differ, so unlike the lane-op families these don't share a
/// `shape.suffix` form). `a`/result are `v128`. `trunc_sat` is the non-trapping float→int (NaN→0,
/// clamp to the integer range); `demote`/`promote` change float width (low 2 lanes, high zeroed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VCvtOp {
    /// `f32x4.convert_i32x4_s`: each `i32` lane → `f32`.
    F32x4ConvertI32x4S,
    /// `f32x4.convert_i32x4_u`: each `u32` lane → `f32`.
    F32x4ConvertI32x4U,
    /// `i32x4.trunc_sat_f32x4_s`: each `f32` lane → saturating `i32`.
    I32x4TruncSatF32x4S,
    /// `i32x4.trunc_sat_f32x4_u`: each `f32` lane → saturating `u32`.
    I32x4TruncSatF32x4U,
    /// `f32x4.demote_f64x2_zero`: the two `f64` lanes → `f32` (lanes 0/1); lanes 2/3 = 0.
    F32x4DemoteF64x2Zero,
    /// `f64x2.promote_low_f32x4`: the low two `f32` lanes → `f64`.
    F64x2PromoteLowF32x4,
}

impl VCvtOp {
    pub const ALL: [VCvtOp; 6] = [
        VCvtOp::F32x4ConvertI32x4S,
        VCvtOp::F32x4ConvertI32x4U,
        VCvtOp::I32x4TruncSatF32x4S,
        VCvtOp::I32x4TruncSatF32x4U,
        VCvtOp::F32x4DemoteF64x2Zero,
        VCvtOp::F64x2PromoteLowF32x4,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VCvtOp::F32x4ConvertI32x4S => "f32x4.convert_i32x4_s",
            VCvtOp::F32x4ConvertI32x4U => "f32x4.convert_i32x4_u",
            VCvtOp::I32x4TruncSatF32x4S => "i32x4.trunc_sat_f32x4_s",
            VCvtOp::I32x4TruncSatF32x4U => "i32x4.trunc_sat_f32x4_u",
            VCvtOp::F32x4DemoteF64x2Zero => "f32x4.demote_f64x2_zero",
            VCvtOp::F64x2PromoteLowF32x4 => "f64x2.promote_low_f32x4",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VCvtOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VCvtOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise **pseudo** min/max on a float `v128` (§17). Unlike the IEEE [`VFloatBinOp::Min`]/`Max`,
/// these are the wasm `pmin`/`pmax`: a plain compare-and-select — `pmin(a,b) = b < a ? b : a`,
/// `pmax(a,b) = a < b ? b : a` — so a NaN operand (and `±0`) follow the select, not IEEE rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VPMinMaxOp {
    Pmin,
    Pmax,
}

impl VPMinMaxOp {
    pub const ALL: [VPMinMaxOp; 2] = [VPMinMaxOp::Pmin, VPMinMaxOp::Pmax];
    pub fn name(self) -> &'static str {
        match self {
            VPMinMaxOp::Pmin => "pmin",
            VPMinMaxOp::Pmax => "pmax",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VPMinMaxOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VPMinMaxOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise binary float ops on a `v128` (§17, IEEE 754, no traps). `Min`/`Max` are the
/// IEEE `minimum`/`maximum` (NaN-propagating, `-0 < +0`) matching the scalar [`FBinOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

impl VFloatBinOp {
    pub const ALL: [VFloatBinOp; 6] = [
        VFloatBinOp::Add,
        VFloatBinOp::Sub,
        VFloatBinOp::Mul,
        VFloatBinOp::Div,
        VFloatBinOp::Min,
        VFloatBinOp::Max,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VFloatBinOp::Add => "add",
            VFloatBinOp::Sub => "sub",
            VFloatBinOp::Mul => "mul",
            VFloatBinOp::Div => "div",
            VFloatBinOp::Min => "min",
            VFloatBinOp::Max => "max",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFloatBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFloatBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Lane-wise unary float ops on a `v128` (§17, IEEE 754, no traps).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VFloatUnOp {
    Abs,
    Neg,
    Sqrt,
}

impl VFloatUnOp {
    pub const ALL: [VFloatUnOp; 3] = [VFloatUnOp::Abs, VFloatUnOp::Neg, VFloatUnOp::Sqrt];
    pub fn name(self) -> &'static str {
        match self {
            VFloatUnOp::Abs => "abs",
            VFloatUnOp::Neg => "neg",
            VFloatUnOp::Sqrt => "sqrt",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VFloatUnOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VFloatUnOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
    }
}

/// Whole-vector bitwise binary ops on a `v128` (§17). Shape-agnostic — they operate on
/// all 128 bits regardless of lane interpretation. `AndNot` is `a & !b`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VBitBinOp {
    And,
    Or,
    Xor,
    AndNot,
}

impl VBitBinOp {
    pub const ALL: [VBitBinOp; 4] = [
        VBitBinOp::And,
        VBitBinOp::Or,
        VBitBinOp::Xor,
        VBitBinOp::AndNot,
    ];
    pub fn name(self) -> &'static str {
        match self {
            VBitBinOp::And => "and",
            VBitBinOp::Or => "or",
            VBitBinOp::Xor => "xor",
            VBitBinOp::AndNot => "andnot",
        }
    }
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&o| o == self).unwrap() as u8
    }
    pub fn from_index(i: u8) -> Option<VBitBinOp> {
        Self::ALL.get(i as usize).copied()
    }
    pub fn from_name(s: &str) -> Option<VBitBinOp> {
        Self::ALL.iter().copied().find(|o| o.name() == s)
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
    /// Call a host capability by **name** — the §7 late-binding import. `import` indexes
    /// [`Module::imports`] (which supplies the name); `handle` is the `i32` capability
    /// handle (frontend-supplied, e.g. loaded from the powerbox stash, exactly like
    /// `cap.call`); `args` are the op arguments; `sig` is a self-describing copy of the
    /// import's op signature (mirroring `cap.call`/`call_indirect`, so result counting needs
    /// no module context). It deliberately carries **no** `type_id`/`op`: those are bound at
    /// instantiation by [`resolve_imports`], the host's `name → (type_id, op)` policy, which
    /// rewrites every `CallImport` into a concrete [`Inst::CapCall`]. A `CallImport` that
    /// reaches the verifier or a backend is a fail-closed error (resolution is mandatory).
    CallImport {
        import: u32,
        sig: FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// §7 capability **reflection** (`cap.self.count`): the number of capabilities the calling
    /// **domain** currently holds — the count of live entries in its own handle table. An
    /// always-available, read-only intrinsic (not a handle-gated `cap.call`): reflecting your own
    /// granted set confers no authority (you already hold every one of those handles), so it does
    /// not widen the §9 egress closure. A nested §14 child has its own table, so it sees only its
    /// attenuated carve. Result is `i32`.
    CapSelfCount,
    /// §7 capability **reflection** (`cap.self.get`): the `idx`-th capability the calling domain
    /// holds, as `(handle: i32, type_id: i32)` — the live handle-table entries in slot order, with
    /// `idx` in `[0, cap.self.count)`. Read-only and authority-neutral (it returns a handle the
    /// domain already possesses); an out-of-range `idx` traps (the guest bounds it by the count).
    /// Lets runtime code discover *what* it was granted and obtain the handle to *use* it.
    CapSelfGet {
        idx: ValIdx,
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
    /// §GC (`GC.md`) **conservative root enumeration** (`gc.roots`): scan every fiber of the
    /// domain — parked fibers, resume-chain ancestors, and the calling computation's own live
    /// frames (the op is **call-clobbering**, so the caller's roots are already spilled to its
    /// control stack, exactly like `cont.resume`/`suspend`) — for candidate pointer words that
    /// fall in the half-open guest-window range `[heap_lo, heap_hi)` (`i64` window offsets).
    /// Writes up to `cap` **distinct (deduplicated)** `i64`-width candidate words, ascending,
    /// into guest memory at byte offset `buf`, and yields the **total** number found (`i64`); if
    /// that exceeds `cap` the guest retries with a larger buffer (only the first `cap` are
    /// written). An **ambient introspection op** — authority-neutral like `cap.self` reflection:
    /// every candidate is an in-window word the guest's own heap already encodes, while
    /// out-of-window words (host return addresses, frame pointers, host pointers) are filtered
    /// *inside* the VM and never cross the boundary, so no host layout leaks (GC.md §3, §6).
    /// Implemented on **both backends**: the interpreter scans its reified `Value` frames; the JIT
    /// conservatively walks the live native control stacks of its fibers (parked fibers' saved
    /// extents `[ctx, top)`, the running resume chain, and the root computation's frames). The two
    /// over-approximate differently (a sound superset of the live roots, not a matching set —
    /// GC.md §3.2). Where the stack-switch substrate is absent the JIT bails `Unsupported` and the
    /// interpreter covers it.
    GcRoots {
        heap_lo: ValIdx,
        heap_hi: ValIdx,
        buf: ValIdx,
        cap: ValIdx,
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
    /// §12 futex notify (`atomic.notify`): wake up to `count` vCPUs waiting on the confined address
    /// `addr`. The count is the **unsigned** "wake up to N" bound (wasm's `memory.atomic.notify` count
    /// is u32; the wake-all idiom is `-1` = `u32::MAX`), so the runtime reinterprets the `i32` bits as
    /// u32 and caps the result at the real waiter count. Yields an `i32`: the number woken. Accesses no
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

    // ----- §17 SIMD: fixed-128 `v128` (D58) -----
    /// `v128.const`: materialize a 16-byte vector constant (little-endian byte order).
    ConstV128([u8; 16]),
    /// `v128.load`: read 16 little-endian bytes from the confined effective address
    /// `addr + offset` into a `v128`. The single widened (16-byte) masked access — the
    /// only escape-TCB delta SIMD adds (§17/D58); confinement masking is implicit, as for
    /// [`Inst::Load`]. `align` is a hint (see [`Inst::Load`]).
    V128Load {
        addr: ValIdx,
        offset: u64,
        align: u8,
    },
    /// `v128.store`: write the 16 little-endian bytes of `value` at the confined effective
    /// address. Produces no SSA result (like [`Inst::Store`]).
    V128Store {
        addr: ValIdx,
        value: ValIdx,
        offset: u64,
        align: u8,
    },
    /// `<shape>.splat`: broadcast a scalar (the shape's [`VShape::lane_val`] type) into
    /// every lane, producing a `v128`.
    Splat {
        shape: VShape,
        a: ValIdx,
    },
    /// `<shape>.extract_lane <lane>`: read lane `lane` of `a` as the shape's scalar type.
    /// For narrow integer shapes (`i8x16`/`i16x8`) `signed` selects sign- vs zero-extension
    /// into the `i32` result; it is ignored for the other shapes.
    ExtractLane {
        shape: VShape,
        lane: u8,
        signed: bool,
        a: ValIdx,
    },
    /// `<shape>.replace_lane <lane>`: `a` with lane `lane` set to scalar `b` (the shape's
    /// [`VShape::lane_val`] type); result `v128`.
    ReplaceLane {
        shape: VShape,
        lane: u8,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise binary integer op (see [`VIntBinOp`]); `a`/`b`/result are `v128`.
    VIntBin {
        shape: VShape,
        op: VIntBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise integer comparison (see [`VICmpOp`]); `a`/`b`/result are `v128` (per-lane all-ones
    /// or all-zeros mask of the lane width).
    VIntCmp {
        shape: VShape,
        op: VICmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise float comparison (see [`VFCmpOp`]); `a`/`b`/result are `v128` (per-lane all-ones or
    /// all-zeros mask of the lane width).
    VFloatCmp {
        shape: VShape,
        op: VFCmpOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise integer shift by a scalar amount (see [`VShiftOp`]): `a`/result are `v128`, `amt`
    /// is an `i32` (taken modulo the lane bit-width).
    VShift {
        shape: VShape,
        op: VShiftOp,
        a: ValIdx,
        amt: ValIdx,
    },
    /// Lane-wise unary integer op (see [`VIntUnOp`]); `a`/result are `v128`.
    VIntUn {
        shape: VShape,
        op: VIntUnOp,
        a: ValIdx,
    },
    /// Lane-wise saturating add/sub (see [`VSatBinOp`]); `a`/`b`/result are `v128`. `i8x16`/`i16x8`
    /// only (verifier-enforced).
    VSatBin {
        shape: VShape,
        op: VSatBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane **widen** (`extend`, see [`VWidenOp`]); `shape` is the **result** (wider) shape, the
    /// source is its [`VShape::narrower`]. `a`/result are `v128`.
    VWiden {
        shape: VShape,
        op: VWidenOp,
        a: ValIdx,
    },
    /// Lane **narrow** (see [`VNarrowOp`]); `shape` is the **result** (narrow) shape, the source is
    /// its [`VShape::wider`]. `a`/`b`/result are `v128`. `i8x16`/`i16x8` only (verifier-enforced).
    VNarrow {
        shape: VShape,
        op: VNarrowOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane int↔float / float↔float conversion (see [`VCvtOp`]); `a`/result are `v128`.
    VConvert {
        op: VCvtOp,
        a: ValIdx,
    },
    /// Lane-wise float pseudo-min/max (see [`VPMinMaxOp`]); `a`/`b`/result are `v128`. Float shapes.
    VPMinMax {
        shape: VShape,
        op: VPMinMaxOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// `v128.any_true`: `i32` `1` if **any** bit of the 128-bit vector is set, else `0`
    /// (shape-agnostic). `a` is `v128`, result `i32`.
    VAnyTrue {
        a: ValIdx,
    },
    /// `<shape>.all_true`: `i32` `1` if **every** lane (of `shape`) is non-zero, else `0`. `a` is
    /// `v128`, result `i32`.
    VAllTrue {
        shape: VShape,
        a: ValIdx,
    },
    /// `<shape>.bitmask`: gather the **high (sign) bit** of each lane into the low bits of an `i32`
    /// (lane `i` → bit `i`). `a` is `v128`, result `i32`.
    VBitmask {
        shape: VShape,
        a: ValIdx,
    },
    /// Lane-wise binary float op (see [`VFloatBinOp`]); `a`/`b`/result are `v128`.
    VFloatBin {
        shape: VShape,
        op: VFloatBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// Lane-wise unary float op (see [`VFloatUnOp`]); `a`/result are `v128`.
    VFloatUn {
        shape: VShape,
        op: VFloatUnOp,
        a: ValIdx,
    },
    /// Whole-vector bitwise binary op (see [`VBitBinOp`]); `a`/`b`/result are `v128`.
    VBitBin {
        op: VBitBinOp,
        a: ValIdx,
        b: ValIdx,
    },
    /// `v128.not`: bitwise complement of all 128 bits.
    VNot {
        a: ValIdx,
    },
    /// `v128.bitselect`: per-bit `(a & mask) | (b & !mask)`. All three operands `v128`.
    Bitselect {
        a: ValIdx,
        b: ValIdx,
        mask: ValIdx,
    },
    /// `i8x16.shuffle`: a constant byte shuffle. Each `lanes[i]` (0..32) selects byte `i`
    /// of the result from the 32-byte concatenation `a ++ b` (indices 0..16 = `a`, 16..32
    /// = `b`). Out-of-range indices (≥32) are verifier-rejected.
    Shuffle {
        lanes: [u8; 16],
        a: ValIdx,
        b: ValIdx,
    },
    /// `i8x16.swizzle`: dynamic byte select — result byte `i` is `a[b[i]]` when `b[i] < 16`,
    /// else `0`. Both operands and result `v128`.
    Swizzle {
        a: ValIdx,
        b: ValIdx,
    },
    /// `simd.width_bytes`: the host's supported SIMD vector width in bytes, as an `i32`.
    /// The §17/D58 **feature-detection hook**. In the fixed-128 MVP this is the constant
    /// `16` on every backend (so it stays deterministic across the interp↔JIT oracle); it
    /// becomes a real runtime query when feature-detected wider widths (`v256`/`v512`) land.
    SimdWidthBytes,
}

impl Inst {
    /// How many values this instruction appends at the next block-local indices.
    ///
    /// Most instructions append exactly one; `Store` appends none; a `Call` appends
    /// its callee's result count, so it needs the per-function result arities
    /// (indexed by [`FuncIdx`]) to answer; `CallIndirect` carries its own signature.
    pub fn result_count(&self, fn_results: &[usize]) -> usize {
        match self {
            Inst::Store { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicFence { .. }
            | Inst::V128Store { .. } => 0,
            // `cont.resume` is the one multi-result non-call op: `(status, value)`.
            Inst::ContResume { .. } => 2,
            // `cap.self.get` appends `(handle, type_id)`; `cap.self.count` appends one `i32`.
            Inst::CapSelfGet { .. } => 2,
            Inst::CapSelfCount => 1,
            Inst::Call { func, .. } => fn_results.get(*func as usize).copied().unwrap_or(0),
            Inst::CallIndirect { ty, .. } => ty.results.len(),
            Inst::CapCall { sig, .. } => sig.results.len(),
            Inst::CallImport { sig, .. } => sig.results.len(),
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

impl Func {
    /// Whether this function uses any §12 fiber/thread/futex op (`cont.*`, `thread.*`,
    /// `atomic.wait`/`notify`). The single source of truth for backends that must agree on
    /// rejecting concurrency in a context that cannot host it — e.g. a §14 JIT child (no
    /// per-child runtimes) or a guest-submitted `Jit`-capability unit (the single-threaded
    /// MVP restriction; DESIGN.md §22 "Concurrency") — so the reference interpreter and the JIT
    /// fail-close on exactly the same set.
    pub fn uses_concurrency(&self) -> bool {
        self.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::ContNew { .. }
                        | Inst::ContResume { .. }
                        | Inst::Suspend { .. }
                        | Inst::ThreadSpawn { .. }
                        | Inst::ThreadJoin { .. }
                        | Inst::MemoryWait { .. }
                        | Inst::MemoryNotify { .. }
                )
            })
        })
    }
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
    /// Named capability imports (§7 "Host-defined capabilities & discoverability"): the
    /// interfaces this module expects the host to bind. Each is a `name` + op `sig`; a
    /// [`Inst::CallImport`] references one by index. The host's instantiation policy resolves
    /// each name to a concrete `(type_id, op)` via [`resolve_imports`], which lowers every
    /// `CallImport` to a [`Inst::CapCall`] and clears this list — so the verifier and both
    /// backends only ever see import-free modules. An import-bearing module that reaches a
    /// backend is a fail-closed error (resolution is mandatory first). Empty for modules that
    /// inline their capability calls (the legacy `cap.call`-only form).
    pub imports: Vec<Import>,
    /// **Debug info — the frontend-neutral waist** (`DEBUGGING.md` §6 / D-DBG-7). Strippable
    /// tooling, **untrusted for escape** (§2a): the verifier never reads it and neither backend's
    /// safety depends on it; `None` ⇒ no debug info, zero cost. Populated by a frontend *during
    /// lowering* (only it knows which source produced which op); consumed host-side by the
    /// interpreter debugger and (later) DWARF/DAP. Slice 1 carries the neutral core (source
    /// locations + variables); the per-producer rich blob is a later field.
    pub debug_info: Option<DebugInfo>,
}

/// The neutral core of the debug-info waist (`DEBUGGING.md` §6): everything the interpreter
/// stepper and backtraces need, in a form **every** frontend can populate (chibicc tokens, LLVM
/// `!DILocation`/`dbg.value`, wasm DWARF). Positions key on `(func, block, inst)` — module 0, the
/// guest's own program (installed §22 units have no source). Format-specific richness (full DWARF
/// DIEs / LLVM DI) is a later opaque per-producer blob the middle never parses.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DebugInfo {
    /// Source file paths, referenced by index from [`Loc::file`].
    pub files: Vec<String>,
    /// Source location of individual ops. An op with no entry inherits nothing (unmapped).
    pub locs: Vec<Loc>,
    /// Source variables and where their value lives (the §6 neutral `VarLoc` = S2).
    pub vars: Vec<VarInfo>,
}

/// A source location for one op (`DEBUGGING.md` §6 neutral core). `col == 0` means "no column"
/// (wasm DWARF often omits columns).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Loc {
    pub func: u32,
    pub block: u32,
    pub inst: u32,
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

/// A source variable and its value location (`DEBUGGING.md` §6 / S2). `ty` is a neutral render
/// name for slice 1 (a structured `TypeRef` — encoding + size + a rich-blob handle — is a later
/// refinement). Function-scoped for slice 1; lexical `IrPc`-range scopes are a later refinement.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VarInfo {
    pub func: u32,
    pub name: String,
    pub ty: String,
    pub loc: VarLoc,
}

/// Where a source variable's value lives at runtime (the S2 value-location model, IR form). The
/// `Machine` (Cranelift register/stack) variant for debugging JIT-optimized code is a later field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VarLoc {
    /// An address-taken / aggregate / narrow local: window data-stack slot at `data-SP + off`.
    Window { off: i64 },
    /// A promoted scalar: the SSA value index that holds it (resolved directly from the frame's
    /// values by the interpreter — no debug-build mode needed).
    Ssa { value: u32 },
}

/// A named capability import (§7). `name` is the symbolic tag the host resolves at
/// instantiation; `sig` is the operation signature (op args → results, excluding the
/// handle). Declared in [`Module::imports`] and referenced by index from
/// [`Inst::CallImport`]. Listing a module's imports is the up-front, fail-closed
/// "what capabilities does this need?" check (a missing binding never silently no-ops).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Import {
    pub name: String,
    pub sig: FuncType,
}

/// A capability binding resolved from an import name at instantiation (§7): the concrete
/// interface `type_id` and operation `op` the host bound the name to. Returned by the
/// resolver passed to [`resolve_imports`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedCap {
    pub type_id: u32,
    pub op: u32,
}

/// What an import **name** binds to when [`resolve_imports_with`] lowers it. The §7 capability
/// case (`Cap`) is the host-ABI binding; `Func` is the **compile-time (static) linking** case — the
/// name resolved to a concrete function index, so the call lowers to a direct [`Inst::Call`]. (A
/// data-symbol binding — lowering to a constant window offset — is a natural follow-up.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolved {
    /// A host capability: lower to a `cap.call` on the import's handle operand (§7).
    Cap(ResolvedCap),
    /// Another function in the **same** linked module (by index): lower to a direct `call`. The
    /// static-linking case — a symbol resolved to a function merged into this module at link time.
    Func(FuncIdx),
    /// A function reached through the shared `call_indirect` **table slot** (the *dynamic*-linking
    /// case): lower to `call_indirect <slot>`, so a separately-compiled unit can call a function it
    /// doesn't share an index space with (e.g. a plugin calling the host program it was loaded into).
    /// The import's handle operand must be a `ConstI32` placeholder — it is patched to `slot` and
    /// reused as the `call_indirect` index (a 1:1 rewrite, no value renumbering).
    Slot(u32),
}

impl From<ResolvedCap> for Resolved {
    fn from(c: ResolvedCap) -> Self {
        Resolved::Cap(c)
    }
}

/// Why [`resolve_imports`] failed (fail-closed: a missing/garbled import never silently
/// becomes a no-op or a wrong call).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ImportError {
    /// The host policy returned no binding for this import name.
    Unresolved(String),
    /// A `CallImport` referenced an import index past the module's [`Module::imports`].
    BadImportIndex(u32),
    /// A [`Resolved::Slot`] binding's import had a handle operand that is **not** a `ConstI32`
    /// placeholder (the frontend must emit one for a dynamic/slot import, since it is patched to the
    /// slot and reused as the `call_indirect` index).
    SlotHandleNotConst,
}

/// Lower every [`Inst::CallImport`] to a concrete [`Inst::CapCall`] using `resolve` (the
/// host's `name → (type_id, op)` instantiation policy, §7), then drop the import section.
/// This is the §7 **late binding at instantiation**: the module names the capabilities it
/// wants, the host decides what each name binds to, and the result is an import-free module
/// the verifier and both backends accept unchanged (so the §3c handle table + use-site
/// checks carry safety — resolution moves only the `(type_id, op)` immediates, never a
/// handle). Fails closed if any declared import name is unresolved, so a missing required
/// capability surfaces at load, never as a silent miscompile.
pub fn resolve_imports(
    module: &Module,
    mut resolve: impl FnMut(&str) -> Option<ResolvedCap>,
) -> Result<Module, ImportError> {
    resolve_imports_with(module, |name| resolve(name).map(Resolved::Cap))
}

/// The general §7 import-lowering pass: like [`resolve_imports`], but a name may bind to **either**
/// a host capability ([`Resolved::Cap`] → `cap.call`) **or** another function in the linked module
/// ([`Resolved::Func`] → a direct `call`). The latter is the compile-time (static) linking step the
/// in-window loader builds on: a symbol resolved to a concrete function index. Each `CallImport`
/// rewrites **1:1** (no value renumbering) — a `Func` binding drops the unused handle operand.
/// Fails closed on an unresolved name. The result is import-free (verifier/both backends accept it).
pub fn resolve_imports_with(
    module: &Module,
    mut resolve: impl FnMut(&str) -> Option<Resolved>,
) -> Result<Module, ImportError> {
    // Resolve each declared import once, up front (so a name binds consistently and a
    // missing one fails before any rewriting).
    let bound: Vec<Resolved> = module
        .imports
        .iter()
        .map(|imp| resolve(&imp.name).ok_or_else(|| ImportError::Unresolved(imp.name.clone())))
        .collect::<Result<_, _>>()?;
    let mut out = module.clone();
    let fn_results: Vec<usize> = out.funcs.iter().map(|f| f.results.len()).collect();
    for f in &mut out.funcs {
        for b in &mut f.blocks {
            // Map each value index to its defining instruction (block params → `None`) — a `Slot`
            // import patches the `ConstI32` that defines its handle operand, so we may need to reach
            // a *different* instruction than the `CallImport` we're rewriting.
            let mut def_of: Vec<Option<usize>> = vec![None; b.params.len()];
            for (p, inst) in b.insts.iter().enumerate() {
                for _ in 0..inst.result_count(&fn_results) {
                    def_of.push(Some(p));
                }
            }
            for i in 0..b.insts.len() {
                let (import, handle) = match &b.insts[i] {
                    Inst::CallImport { import, handle, .. } => (*import, *handle),
                    _ => continue,
                };
                let bind = *bound
                    .get(import as usize)
                    .ok_or(ImportError::BadImportIndex(import))?;
                // Pull the call's pieces out of the placeholder so we can rebuild it.
                let (sig, args) = match &mut b.insts[i] {
                    Inst::CallImport { sig, args, .. } => {
                        (std::mem::take(sig), std::mem::take(args))
                    }
                    _ => unreachable!(),
                };
                b.insts[i] = match bind {
                    Resolved::Cap(cap) => Inst::CapCall {
                        type_id: cap.type_id,
                        op: cap.op,
                        sig,
                        handle,
                        args,
                    },
                    // Static-link a function symbol → a direct call (handle unused; sig re-checked).
                    Resolved::Func(func) => Inst::Call { func, args },
                    // Dynamic-link a function symbol → a `call_indirect` through the table slot: patch
                    // the handle's `ConstI32` placeholder to `slot` and reuse it as the index.
                    Resolved::Slot(slot) => {
                        let def = def_of
                            .get(handle as usize)
                            .copied()
                            .flatten()
                            .ok_or(ImportError::SlotHandleNotConst)?;
                        match &mut b.insts[def] {
                            Inst::ConstI32(c) => *c = slot as i32,
                            _ => return Err(ImportError::SlotHandleNotConst),
                        }
                        Inst::CallIndirect {
                            ty: sig,
                            idx: handle,
                            args,
                        }
                    }
                };
            }
        }
    }
    out.imports.clear();
    Ok(out)
}

/// One unit to statically link: a module plus the symbols it **exports** and the **relocations** its
/// own data references need. Function symbols (`exports`) and its named `Module::imports` are resolved
/// against the other units by [`link`]; `data_exports` name window offsets within the unit's data; and
/// `relocations` patch the unit's address constants once its data is placed (see [`DataReloc`]).
#[derive(Clone, Debug, Default)]
pub struct LinkUnit {
    pub module: Module,
    /// Function symbols this unit provides: `name → local function index`.
    pub exports: Vec<(String, FuncIdx)>,
    /// Data symbols this unit provides: `name → byte offset within the unit's (un-relocated) data`.
    pub data_exports: Vec<(String, u64)>,
    /// Relocations the unit's data references need after the linker places its data (§ELF-style).
    pub relocations: Vec<DataReloc>,
}

/// A relocation: at link time, patch the **constant** at `(func, block, inst)` — which must be a
/// `ConstI64`/`ConstI32` (an address the frontend left at a unit-local value) — by **adding** a base
/// resolved from `kind`; the const's current value is the **addend** (so `&g + 4` works). This is how
/// a unit's data references survive relocation into the one shared window — no IR change, no value
/// renumbering, just a const edit. `func`/`block`/`inst` are unit-local (applied before concatenation).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataReloc {
    pub func: u32,
    pub block: u32,
    pub inst: u32,
    pub kind: RelocKind,
}

/// What base a [`DataReloc`] adds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RelocKind {
    /// This unit's own assigned data base — the const holds a unit-local data offset (an own-data ref).
    SelfData,
    /// The resolved address of an exported **data** symbol (a cross-unit data import).
    DataSymbol(String),
}

/// Why [`link`] failed (fail-closed; the linked module is also re-verified before it runs).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LinkError {
    /// Two units export the same symbol.
    DuplicateSymbol(String),
    /// An export names a function index past its unit's `funcs`.
    BadExport { symbol: String, index: FuncIdx },
    /// A unit imports a symbol no unit exports.
    Unresolved(String),
    /// A `CallImport` referenced an out-of-range import (a malformed unit).
    BadImportIndex(u32),
    /// A relocation pointed at a missing instruction, or one that isn't an address constant.
    BadReloc(DataReloc),
}

/// **Statically link** units into one module — the compile-time loader (dynamic-linking milestones
/// 1–2). Concatenate the units' functions into one list, **reindexing** each unit's internal function
/// references by its base offset; place each unit's **data** in a non-overlapping window region and
/// apply its **relocations** so its address constants follow; build function + data symbol tables from
/// all exports; and resolve every unit's named imports — a `call` symbol to a **direct call**, a data
/// symbol to a **constant address** — against them. The result is one import-free, relocated module —
/// re-verify it before running, since a unit is untrusted like any frontend output (a cross-unit
/// signature mismatch is caught there).
pub fn link(units: &[LinkUnit]) -> Result<Module, LinkError> {
    // Function and data layout: each unit's functions occupy `[fbase, fbase + n_funcs)` in the merged
    // list, and its data occupies the window region `[dbase, dbase + data_span)` (16-byte aligned, so
    // units never overlap). `data_span` is the high-water mark of the unit's own (un-relocated) data.
    let align16 = |x: u64| (x + 15) & !15;
    let mut fbases = Vec::with_capacity(units.len());
    let mut dbases = Vec::with_capacity(units.len());
    let (mut ftotal, mut dtotal): (u32, u64) = (0, 0);
    for u in units {
        fbases.push(ftotal);
        ftotal += u.module.funcs.len() as u32;
        let dbase = align16(dtotal);
        dbases.push(dbase);
        let span = u
            .module
            .data
            .iter()
            .map(|d| d.offset + d.bytes.len() as u64)
            .max()
            .unwrap_or(0);
        dtotal = dbase + span;
    }
    // Symbol tables: exported name → global function index, and exported data name → window address.
    let mut funcs_tab: std::collections::HashMap<String, FuncIdx> =
        std::collections::HashMap::new();
    let mut data_tab: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (u, (&fbase, &dbase)) in units.iter().zip(fbases.iter().zip(&dbases)) {
        for (name, local) in &u.exports {
            if *local as usize >= u.module.funcs.len() {
                return Err(LinkError::BadExport {
                    symbol: name.clone(),
                    index: *local,
                });
            }
            if funcs_tab.insert(name.clone(), fbase + local).is_some()
                || data_tab.contains_key(name)
            {
                return Err(LinkError::DuplicateSymbol(name.clone()));
            }
        }
        for (name, local_off) in &u.data_exports {
            if data_tab.insert(name.clone(), dbase + local_off).is_some()
                || funcs_tab.contains_key(name)
            {
                return Err(LinkError::DuplicateSymbol(name.clone()));
            }
        }
    }
    // Per unit: place its data, apply its relocations, reindex its functions, resolve its imports.
    let mut funcs: Vec<Func> = Vec::with_capacity(ftotal as usize);
    let mut data: Vec<Data> = Vec::new();
    for (u, (&fbase, &dbase)) in units.iter().zip(fbases.iter().zip(&dbases)) {
        let mut m = u.module.clone();
        // Relocate this unit's data segments into its assigned window region…
        for d in &mut m.data {
            d.offset += dbase;
        }
        // …and patch its address constants so they point at the relocated data (own or imported).
        for r in &u.relocations {
            let base = match &r.kind {
                RelocKind::SelfData => dbase,
                RelocKind::DataSymbol(name) => *data_tab
                    .get(name)
                    .ok_or_else(|| LinkError::Unresolved(name.clone()))?,
            };
            apply_reloc(&mut m, r, base)?;
        }
        offset_func_indices(&mut m, fbase);
        let resolved =
            resolve_imports_with(&m, |name| funcs_tab.get(name).copied().map(Resolved::Func))
                .map_err(|e| match e {
                    ImportError::Unresolved(n) => LinkError::Unresolved(n),
                    ImportError::BadImportIndex(i) => LinkError::BadImportIndex(i),
                    // `link` only ever resolves to `Func` (static merge), never `Slot`, so a
                    // slot-handle error cannot arise here.
                    ImportError::SlotHandleNotConst => {
                        unreachable!("static link never resolves to a Slot")
                    }
                })?;
        funcs.extend(resolved.funcs);
        data.extend(resolved.data);
    }
    Ok(Module {
        funcs,
        // The merged window is the largest any unit declared (they share one linear memory).
        memory: units
            .iter()
            .filter_map(|u| u.module.memory)
            .max_by_key(|m| m.size_log2),
        data,
        imports: Vec::new(),
        // Merging per-unit debug info (with the reindexed function indices) is a follow-up.
        debug_info: None,
    })
}

/// Apply one [`DataReloc`]: add `base` to the addend held in the constant it points at (a `ConstI64`/
/// `ConstI32`). A reloc pointing at a missing or non-constant instruction is fail-closed.
fn apply_reloc(m: &mut Module, r: &DataReloc, base: u64) -> Result<(), LinkError> {
    let inst = m
        .funcs
        .get_mut(r.func as usize)
        .and_then(|f| f.blocks.get_mut(r.block as usize))
        .and_then(|b| b.insts.get_mut(r.inst as usize))
        .ok_or_else(|| LinkError::BadReloc(r.clone()))?;
    match inst {
        Inst::ConstI64(c) => *c = c.wrapping_add(base as i64),
        Inst::ConstI32(c) => *c = (*c as i64).wrapping_add(base as i64) as i32,
        _ => return Err(LinkError::BadReloc(r.clone())),
    }
    Ok(())
}

/// Add `offset` to every **static function index** in `m` (the merged-module reindex): `call`,
/// `ref.func`, `thread.spawn`, and the `return_call` terminator. `call_indirect`/`cont.*` dispatch on
/// runtime funcref *values*, not static indices, so they are untouched. `call.import` carries an
/// import index (not a function index) and is likewise untouched — it is resolved separately.
fn offset_func_indices(m: &mut Module, offset: u32) {
    if offset == 0 {
        return;
    }
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                match inst {
                    Inst::Call { func, .. }
                    | Inst::RefFunc { func }
                    | Inst::ThreadSpawn { func, .. } => *func += offset,
                    _ => {}
                }
            }
            if let Terminator::ReturnCall { func, .. } = &mut b.term {
                *func += offset;
            }
        }
    }
}

/// An initialized data segment (§3a / D40). Placed in the window `[offset, offset+bytes.len())`
/// at instantiation; `readonly` ones are protected after the copy so guest writes fault.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Data {
    pub offset: u64,
    pub readonly: bool,
    pub bytes: Vec<u8>,
}

#[cfg(test)]
mod import_tests {
    use super::*;

    // Build a one-function module whose body issues two CallImports ("write", "exit").
    fn module_with_imports() -> Module {
        let sig_write = FuncType {
            params: vec![ValType::I64, ValType::I64],
            results: vec![ValType::I64],
        };
        let sig_exit = FuncType {
            params: vec![ValType::I32],
            results: vec![],
        };
        let block = Block {
            params: vec![ValType::I32], // v0 = a capability handle
            insts: vec![
                Inst::ConstI64(0), // v1 = buf
                Inst::ConstI64(3), // v2 = len
                Inst::CallImport {
                    // v3 = write(handle=v0, v1, v2)
                    import: 0,
                    sig: sig_write.clone(),
                    handle: 0,
                    args: vec![1, 2],
                },
                Inst::ConstI32(0), // v4 = exit code
                Inst::CallImport {
                    // exit(handle=v0, v4)
                    import: 1,
                    sig: sig_exit.clone(),
                    handle: 0,
                    args: vec![4],
                },
            ],
            term: Terminator::Unreachable,
        };
        Module {
            funcs: vec![Func {
                params: vec![ValType::I32],
                results: vec![],
                blocks: vec![block],
            }],
            memory: None,
            data: vec![],
            imports: vec![
                Import {
                    name: "write".into(),
                    sig: sig_write,
                },
                Import {
                    name: "exit".into(),
                    sig: sig_exit,
                },
            ],
            debug_info: None,
        }
    }

    // The host policy under test: "write" → (Stream=0, op 1), "exit" → (Exit=1, op 0).
    fn policy(name: &str) -> Option<ResolvedCap> {
        match name {
            "write" => Some(ResolvedCap { type_id: 0, op: 1 }),
            "exit" => Some(ResolvedCap { type_id: 1, op: 0 }),
            _ => None,
        }
    }

    #[test]
    fn resolves_callimports_to_capcalls() {
        let m = module_with_imports();
        let r = resolve_imports(&m, policy).expect("resolve");
        // Import section is gone; the module is now backend-ready.
        assert!(r.imports.is_empty());
        let insts = &r.funcs[0].blocks[0].insts;
        // No CallImport survives.
        assert!(
            !insts.iter().any(|i| matches!(i, Inst::CallImport { .. })),
            "all imports must be lowered"
        );
        // "write" became cap.call 0 1 on handle v0 with args [1,2].
        match &insts[2] {
            Inst::CapCall {
                type_id,
                op,
                handle,
                args,
                sig,
            } => {
                assert_eq!((*type_id, *op, *handle), (0, 1, 0));
                assert_eq!(args, &vec![1, 2]);
                assert_eq!(sig.results.len(), 1);
            }
            other => panic!("expected CapCall, got {other:?}"),
        }
        // "exit" became cap.call 1 0.
        match &insts[4] {
            Inst::CapCall {
                type_id, op, args, ..
            } => {
                assert_eq!((*type_id, *op), (1, 0));
                assert_eq!(args, &vec![4]);
            }
            other => panic!("expected CapCall, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_import_fails_closed() {
        let m = module_with_imports();
        // A policy that knows "write" but not "exit" must error, not silently drop it.
        let err = resolve_imports(&m, |n| {
            (n == "write").then_some(ResolvedCap { type_id: 0, op: 1 })
        })
        .expect_err("must fail closed");
        assert_eq!(err, ImportError::Unresolved("exit".into()));
    }

    #[test]
    fn module_without_imports_is_unchanged() {
        let mut m = module_with_imports();
        // Replace the import calls with a plain return so there's nothing to resolve.
        m.imports.clear();
        m.funcs[0].blocks[0].insts.clear();
        m.funcs[0].blocks[0].term = Terminator::Return(vec![]);
        let r = resolve_imports(&m, policy).expect("resolve");
        assert_eq!(r, m, "a no-import module round-trips identically");
    }
}
