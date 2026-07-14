//! The executable ISA spec (SPEC.md): one machine-readable op table — typing as data,
//! reference semantics as closures — from which the conformance suites in
//! `crates/svm/tests/spec_*.rs` are generated.
//!
//! **Independence rule (SPEC.md):** the `eval` closures here are written from the
//! `DESIGN.md` §3b prose, never lifted from `svm-interp` — this crate is a *second,
//! independent* statement of the semantics, so a spec↔backend divergence is a bug in
//! one of them, not a tautology. Test-tier only: nothing on the runtime path depends
//! on this crate.
//!
//! Landed slices (SPEC.md implementation plan): the scalar integer core (slice 1),
//! scalar floats + conversions (slice 2), the restated opcode byte map (slice 3,
//! [`OpRow::encoding`]), the reference verifier (slice 4, [`verify`]), and the
//! memory window model + load/store/bulk rows (slice 5, [`mem_rows`]).

#![forbid(unsafe_code)]

pub mod verify;

use svm_ir::*;

/// A spec-level value. Floats are carried as **raw bits** so expectations are bit-exact
/// (`f32`/`f64` `PartialEq` would make `NaN != NaN` and `-0.0 == +0.0`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecVal {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
}

impl SpecVal {
    pub fn ty(self) -> ValType {
        match self {
            SpecVal::I32(_) => ValType::I32,
            SpecVal::I64(_) => ValType::I64,
            SpecVal::F32(_) => ValType::F32,
            SpecVal::F64(_) => ValType::F64,
        }
    }
    /// True for a float value whose bit pattern the IR does *not* pin (§3b: NaN bits are
    /// host-defined in default mode) — conformance compares these as "is a NaN", not bits.
    pub fn is_unpinned_nan(self) -> bool {
        match self {
            SpecVal::F32(b) => f32::from_bits(b).is_nan(),
            SpecVal::F64(b) => f64::from_bits(b).is_nan(),
            _ => false,
        }
    }
}

/// Trap kinds the deterministic core can produce (§3b). Grows with later slices
/// (`MemoryFault` at the memory slice); host/concurrency trap kinds stay out of scope.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecTrap {
    DivByZero,
    IntOverflow,
    BadConversion,
    MemoryFault,
}

/// The op's semantic class (SPEC.md). Drives which suites cover it: `Pure`/`Trapping`
/// rows get semantic vectors; `Memory` rows get the window model (slice 5); `Control`/
/// `Host`/`Concurrency` rows get typing + encoding only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Pure,
    Trapping,
    Memory,
    Control,
    Host,
    Concurrency,
}

/// How a row's inputs reach the op in a generated module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// Inputs arrive as function parameters feeding the op's SSA operands — the module
    /// is input-independent, so a backend compiles it once and runs every vector.
    Operands,
    /// The op takes no SSA operands; the single input is baked as the immediate
    /// (the `const` ops), so each vector is its own module.
    Immediate,
}

/// The wire encoding of an op (SPEC.md suite 3): its primary opcode byte, or the SIMD
/// escape prefix plus sub-opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Enc {
    Byte(u8),
    Prefixed(u8, u8),
}

/// Constructor of a row's op over operand value indices (plus the immediate vector for
/// `Shape::Immediate` rows).
pub type BuildFn = Box<dyn Fn(&[ValIdx], &[SpecVal]) -> Inst>;
/// A row's reference semantics over one input vector.
pub type EvalFn = Box<dyn Fn(&[SpecVal]) -> Result<SpecVal, SpecTrap>>;

/// One row of the op table: a concrete op (variant × sub-op × type) with its typing,
/// class, IR constructor, and reference semantics.
pub struct OpRow {
    /// Text mnemonic, matching `svm-text` exactly (e.g. `"i32.div_s"`).
    pub id: String,
    /// SSA operand types, in order (empty for `Shape::Immediate`).
    pub operands: Vec<ValType>,
    /// Result type. (Every slice-1 op has exactly one result.)
    pub result: ValType,
    pub class: Class,
    pub shape: Shape,
    /// The op's wire encoding, restated independently of `svm-encode` (suite 3).
    pub encoding: Enc,
    /// Construct the op over operand value indices (in `operands` order); an
    /// `Immediate` row reads its immediate from the vector instead.
    pub build: BuildFn,
    /// Reference semantics over one input vector (see `inputs`).
    pub eval: EvalFn,
}

impl OpRow {
    /// The types `eval` consumes and [`vectors_for`] enumerates: the SSA operands, or
    /// the baked immediate for a const row.
    pub fn inputs(&self) -> Vec<ValType> {
        match self.shape {
            Shape::Operands => self.operands.clone(),
            Shape::Immediate => vec![self.result],
        }
    }
}

// --- reference semantics -----------------------------------------------------------
//
// Written from DESIGN.md §3b: two's-complement wrap; `div`/`rem` trap on a zero
// divisor; only signed *division* of INT_MIN/−1 traps (`IntOverflow`) — the remainder
// there is 0, which is representable, so `rem_s` returns 0; shift/rotate amounts are
// taken mod the bitwidth; bit counts return in the operand's own type.

pub fn bin_i32(op: BinOp, a: i32, b: i32) -> Result<i32, SpecTrap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            if a == i32::MIN && b == -1 {
                return Err(SpecTrap::IntOverflow);
            }
            a / b // truncating quotient; the trap cases above are the only UB-shaped inputs
        }
        BinOp::DivU => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            ((a as u32) / (b as u32)) as i32
        }
        BinOp::RemS => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            a.wrapping_rem(b) // INT_MIN % −1 == 0, representable — no trap (§3b)
        }
        BinOp::RemU => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            ((a as u32) % (b as u32)) as i32
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a << (b & 31),
        BinOp::ShrS => a >> (b & 31),
        BinOp::ShrU => ((a as u32) >> (b & 31)) as i32,
        BinOp::Rotl => a.rotate_left((b & 31) as u32),
        BinOp::Rotr => a.rotate_right((b & 31) as u32),
    })
}

pub fn bin_i64(op: BinOp, a: i64, b: i64) -> Result<i64, SpecTrap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            if a == i64::MIN && b == -1 {
                return Err(SpecTrap::IntOverflow);
            }
            a / b
        }
        BinOp::DivU => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            ((a as u64) / (b as u64)) as i64
        }
        BinOp::RemS => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            a.wrapping_rem(b) // INT_MIN % −1 == 0, representable — no trap (§3b)
        }
        BinOp::RemU => {
            if b == 0 {
                return Err(SpecTrap::DivByZero);
            }
            ((a as u64) % (b as u64)) as i64
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a << (b & 63),
        BinOp::ShrS => a >> (b & 63),
        BinOp::ShrU => ((a as u64) >> (b & 63)) as i64,
        BinOp::Rotl => a.rotate_left((b & 63) as u32),
        BinOp::Rotr => a.rotate_right((b & 63) as u32),
    })
}

pub fn cmp_i32(op: CmpOp, a: i32, b: i32) -> i32 {
    let r = match op {
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
    };
    r as i32
}

pub fn cmp_i64(op: CmpOp, a: i64, b: i64) -> i32 {
    let r = match op {
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
    };
    r as i32
}

pub fn un_i32(op: IntUnOp, a: i32) -> i32 {
    match op {
        IntUnOp::Clz => a.leading_zeros() as i32,
        IntUnOp::Ctz => a.trailing_zeros() as i32,
        IntUnOp::Popcnt => a.count_ones() as i32,
        IntUnOp::Extend8S => a as i8 as i32,
        IntUnOp::Extend16S => a as i16 as i32,
        IntUnOp::Extend32S => a, // identity on i32 (§3b)
    }
}

pub fn un_i64(op: IntUnOp, a: i64) -> i64 {
    match op {
        IntUnOp::Clz => a.leading_zeros() as i64,
        IntUnOp::Ctz => a.trailing_zeros() as i64,
        IntUnOp::Popcnt => a.count_ones() as i64,
        IntUnOp::Extend8S => a as i8 as i64,
        IntUnOp::Extend16S => a as i16 as i64,
        IntUnOp::Extend32S => a as i32 as i64,
    }
}

fn as_i32(v: SpecVal) -> i32 {
    match v {
        SpecVal::I32(x) => x,
        _ => unreachable!("spec row fed a non-i32 input"),
    }
}
fn as_i64(v: SpecVal) -> i64 {
    match v {
        SpecVal::I64(x) => x,
        _ => unreachable!("spec row fed a non-i64 input"),
    }
}
fn as_f32(v: SpecVal) -> f32 {
    match v {
        SpecVal::F32(b) => f32::from_bits(b),
        _ => unreachable!("spec row fed a non-f32 input"),
    }
}
fn as_f64(v: SpecVal) -> f64 {
    match v {
        SpecVal::F64(b) => f64::from_bits(b),
        _ => unreachable!("spec row fed a non-f64 input"),
    }
}

/// `CastOp` semantics (§3b): demote/promote are IEEE 754 round-to-nearest-even width
/// changes (a NaN result's bit pattern is unpinned — compared as "is NaN"); the
/// reinterprets move bits between an integer and a float of the same width, exactly.
pub fn cast(op: CastOp, x: SpecVal) -> SpecVal {
    match op {
        CastOp::Demote => SpecVal::F32((as_f64(x) as f32).to_bits()),
        CastOp::Promote => SpecVal::F64((as_f32(x) as f64).to_bits()),
        CastOp::ReinterpI32F32 => SpecVal::F32(as_i32(x) as u32),
        CastOp::ReinterpF32I32 => match x {
            SpecVal::F32(b) => SpecVal::I32(b as i32),
            _ => unreachable!("spec row fed a non-f32 input"),
        },
        CastOp::ReinterpI64F64 => SpecVal::F64(as_i64(x) as u64),
        CastOp::ReinterpF64I64 => match x {
            SpecVal::F64(b) => SpecVal::I64(b as i64),
            _ => unreachable!("spec row fed a non-f64 input"),
        },
    }
}

// --- float reference semantics (slice 2) ---------------------------------------------
//
// DESIGN.md §3b: IEEE 754, round-to-nearest-even, no traps (results go to inf/NaN);
// NaN result bit patterns are unpinned (compared as "is NaN"). Where the prose is
// terse the spec pins the intended (wasm-identical) semantics precisely: `min`/`max`
// propagate NaN and order `-0 < +0`; `nearest` rounds ties to even; `fma` is the
// correctly-rounded fused multiply-add; `abs`/`neg`/`copysign` are pure sign-bit
// operations (defined on NaN too).

/// NaN-propagating minimum with `-0 < +0` (wasm `fmin`; NOT Rust's `f32::min`, which
/// is IEEE minNum — it drops NaN and can't tell the zeros apart).
fn fmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        // Equal compares include -0 == +0: pick the negative-signed one.
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

/// NaN-propagating maximum with `-0 < +0` (wasm `fmax`); see [`fmin`].
fn fmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else if a > b {
        a
    } else {
        b
    }
}

pub fn fbin_f32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b, // /0 → ±inf, 0/0 → NaN — floats never trap (§3b)
        FBinOp::Min => fmin(a as f64, b as f64) as f32, // exact: f32 ⊂ f64, order-only
        FBinOp::Max => fmax(a as f64, b as f64) as f32,
        FBinOp::Copysign => a.copysign(b),
    }
}

pub fn fbin_f64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin(a, b),
        FBinOp::Max => fmax(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

pub fn fun_f32(op: FUnOp, a: f32) -> f32 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(), // IEEE correctly-rounded; sqrt(neg) → NaN
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

pub fn fun_f64(op: FUnOp, a: f64) -> f64 {
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

/// IEEE partial-order comparison: any comparison with a NaN is false, so `ne` (the
/// negation of `eq`) is *true* on NaN.
pub fn fcmp(op: FCmpOp, a: f64, b: f64) -> i32 {
    let r = match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    };
    r as i32
}

/// Saturating float→int (`trunc_sat`, the §3b default): truncate toward zero; NaN → 0;
/// out-of-range saturates to the target's MIN/MAX. (Rust's float→int `as` implements
/// exactly this definition.)
pub fn trunc_sat(op: FToI, x: SpecVal) -> SpecVal {
    // Promoting f32 → f64 is exact, so the eight cases collapse to four on f64.
    let f: f64 = match op.parts().0 {
        FloatTy::F32 => as_f32(x) as f64,
        FloatTy::F64 => as_f64(x),
    };
    match op.parts() {
        (_, IntTy::I32, true) => SpecVal::I32(f as i32),
        (_, IntTy::I32, false) => SpecVal::I32(f as u32 as i32),
        (_, IntTy::I64, true) => SpecVal::I64(f as i64),
        (_, IntTy::I64, false) => SpecVal::I64(f as u64 as i64),
    }
}

/// Trapping float→int (`trunc`): NaN and out-of-range trap `BadConversion` instead of
/// saturating. In range ⇔ the truncation toward zero is representable — i.e. `x` lies
/// strictly inside `(MIN−1, MAX+1)`, stated with the exact float boundary constants
/// (`2^31`, `2^32`, `2^63`, `2^64` and their negatives are all exact in f64; the i64
/// signed lower bound is closed because `−2^63−1` is not representable in f64).
pub fn trunc_trap(op: FToI, x: SpecVal) -> Result<SpecVal, SpecTrap> {
    let f: f64 = match op.parts().0 {
        FloatTy::F32 => as_f32(x) as f64,
        FloatTy::F64 => as_f64(x),
    };
    if f.is_nan() {
        return Err(SpecTrap::BadConversion);
    }
    let in_range = match op.parts() {
        (_, IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (_, IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (_, IntTy::I64, true) => {
            (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&f)
        }
        (_, IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return Err(SpecTrap::BadConversion);
    }
    Ok(trunc_sat(op, x)) // in range, so saturation never engages — plain truncation
}

/// Int→float (`convert`): round-to-nearest-even to the target width (Rust's int→float
/// `as` implements exactly this).
pub fn itof(op: IToF, x: SpecVal) -> SpecVal {
    match op.parts() {
        (IntTy::I32, FloatTy::F32, true) => SpecVal::F32((as_i32(x) as f32).to_bits()),
        (IntTy::I32, FloatTy::F32, false) => SpecVal::F32((as_i32(x) as u32 as f32).to_bits()),
        (IntTy::I64, FloatTy::F32, true) => SpecVal::F32((as_i64(x) as f32).to_bits()),
        (IntTy::I64, FloatTy::F32, false) => SpecVal::F32((as_i64(x) as u64 as f32).to_bits()),
        (IntTy::I32, FloatTy::F64, true) => SpecVal::F64((as_i32(x) as f64).to_bits()),
        (IntTy::I32, FloatTy::F64, false) => SpecVal::F64((as_i32(x) as u32 as f64).to_bits()),
        (IntTy::I64, FloatTy::F64, true) => SpecVal::F64((as_i64(x) as f64).to_bits()),
        (IntTy::I64, FloatTy::F64, false) => SpecVal::F64((as_i64(x) as u64 as f64).to_bits()),
    }
}

// --- the byte map, restated (SPEC.md suite 3) -----------------------------------------
//
// A second, independent statement of `svm-encode`'s `mod op` opcode map, written as
// **explicit per-op bytes** — NOT `base + op.index()`. Sharing the encoder's own
// `index()` would let a sub-enum reorder shift both sides in lockstep, which is
// exactly the silent format break suite 3 exists to catch. Every match is exhaustive,
// so a new sub-op forces a conscious byte assignment here.

fn enc_int_bin(ty: IntTy, op: BinOp) -> Enc {
    let base: u8 = match ty {
        IntTy::I32 => 0x20,
        IntTy::I64 => 0x40,
    };
    let off: u8 = match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::DivS => 3,
        BinOp::DivU => 4,
        BinOp::RemS => 5,
        BinOp::RemU => 6,
        BinOp::And => 7,
        BinOp::Or => 8,
        BinOp::Xor => 9,
        BinOp::Shl => 10,
        BinOp::ShrS => 11,
        BinOp::ShrU => 12,
        BinOp::Rotl => 13,
        BinOp::Rotr => 14,
    };
    Enc::Byte(base + off)
}

fn enc_int_cmp(ty: IntTy, op: CmpOp) -> Enc {
    let base: u8 = match ty {
        IntTy::I32 => 0x31,
        IntTy::I64 => 0x51,
    };
    let off: u8 = match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::LtS => 2,
        CmpOp::LtU => 3,
        CmpOp::LeS => 4,
        CmpOp::LeU => 5,
        CmpOp::GtS => 6,
        CmpOp::GtU => 7,
        CmpOp::GeS => 8,
        CmpOp::GeU => 9,
    };
    Enc::Byte(base + off)
}

fn enc_int_un(ty: IntTy, op: IntUnOp) -> Enc {
    let base: u8 = match ty {
        IntTy::I32 => 0x14,
        IntTy::I64 => 0x1A,
    };
    let off: u8 = match op {
        IntUnOp::Clz => 0,
        IntUnOp::Ctz => 1,
        IntUnOp::Popcnt => 2,
        IntUnOp::Extend8S => 3,
        IntUnOp::Extend16S => 4,
        IntUnOp::Extend32S => 5,
    };
    Enc::Byte(base + off)
}

fn enc_convert(op: ConvOp) -> Enc {
    Enc::Byte(match op {
        ConvOp::ExtendI32S => 0x60,
        ConvOp::ExtendI32U => 0x61,
        ConvOp::WrapI64 => 0x62,
    })
}

fn enc_cast(op: CastOp) -> Enc {
    Enc::Byte(match op {
        CastOp::Demote => 0xC0,
        CastOp::Promote => 0xC1,
        CastOp::ReinterpI32F32 => 0xC2,
        CastOp::ReinterpF32I32 => 0xC3,
        CastOp::ReinterpI64F64 => 0xC4,
        CastOp::ReinterpF64I64 => 0xC5,
    })
}

fn enc_fbin(ty: FloatTy, op: FBinOp) -> Enc {
    let base: u8 = match ty {
        FloatTy::F32 => 0x90,
        FloatTy::F64 => 0xA0,
    };
    let off: u8 = match op {
        FBinOp::Add => 0,
        FBinOp::Sub => 1,
        FBinOp::Mul => 2,
        FBinOp::Div => 3,
        FBinOp::Min => 4,
        FBinOp::Max => 5,
        FBinOp::Copysign => 6,
    };
    Enc::Byte(base + off)
}

fn enc_fun(ty: FloatTy, op: FUnOp) -> Enc {
    let base: u8 = match ty {
        FloatTy::F32 => 0x98,
        FloatTy::F64 => 0xA8,
    };
    let off: u8 = match op {
        FUnOp::Abs => 0,
        FUnOp::Neg => 1,
        FUnOp::Sqrt => 2,
        FUnOp::Ceil => 3,
        FUnOp::Floor => 4,
        FUnOp::Trunc => 5,
        FUnOp::Nearest => 6,
    };
    Enc::Byte(base + off)
}

fn enc_fcmp(ty: FloatTy, op: FCmpOp) -> Enc {
    let base: u8 = match ty {
        FloatTy::F32 => 0xB0,
        FloatTy::F64 => 0xB8,
    };
    let off: u8 = match op {
        FCmpOp::Eq => 0,
        FCmpOp::Ne => 1,
        FCmpOp::Lt => 2,
        FCmpOp::Le => 3,
        FCmpOp::Gt => 4,
        FCmpOp::Ge => 5,
    };
    Enc::Byte(base + off)
}

/// Offset shared by the saturating (`0xD0+`) and trapping (`0xD8+`) families.
fn ftoi_off(op: FToI) -> u8 {
    match op {
        FToI::F32I32S => 0,
        FToI::F32I32U => 1,
        FToI::F32I64S => 2,
        FToI::F32I64U => 3,
        FToI::F64I32S => 4,
        FToI::F64I32U => 5,
        FToI::F64I64S => 6,
        FToI::F64I64U => 7,
    }
}

fn enc_itof(op: IToF) -> Enc {
    Enc::Byte(match op {
        IToF::I32F32S => 0xE0,
        IToF::I32F32U => 0xE1,
        IToF::I64F32S => 0xE2,
        IToF::I64F32U => 0xE3,
        IToF::I32F64S => 0xE4,
        IToF::I32F64U => 0xE5,
        IToF::I64F64S => 0xE6,
        IToF::I64F64U => 0xE7,
    })
}

// --- the op table -------------------------------------------------------------------

/// Slice-1 rows: the scalar integer core plus `Cast` and the pointer ops. Later slices
/// extend this (floats, memory, SIMD) until the [`coverage`] walk is fully specced.
pub fn scalar_rows() -> Vec<OpRow> {
    let mut rows: Vec<OpRow> = Vec::new();
    let mut push = |row: OpRow| rows.push(row);

    // Consts: the immediate is the single input; the op reproduces it.
    push(OpRow {
        id: "i32.const".into(),
        operands: vec![],
        result: ValType::I32,
        class: Class::Pure,
        shape: Shape::Immediate,
        encoding: Enc::Byte(0x10),
        build: Box::new(|_, imm| Inst::ConstI32(as_i32(imm[0]))),
        eval: Box::new(|x| Ok(x[0])),
    });
    push(OpRow {
        id: "i64.const".into(),
        operands: vec![],
        result: ValType::I64,
        class: Class::Pure,
        shape: Shape::Immediate,
        encoding: Enc::Byte(0x11),
        build: Box::new(|_, imm| Inst::ConstI64(as_i64(imm[0]))),
        eval: Box::new(|x| Ok(x[0])),
    });

    for ty in [IntTy::I32, IntTy::I64] {
        let vt = ty.val();

        for op in BinOp::ALL {
            let class = match op {
                BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => Class::Trapping,
                _ => Class::Pure,
            };
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt, vt],
                result: vt,
                class,
                shape: Shape::Operands,
                encoding: enc_int_bin(ty, op),
                build: Box::new(move |v, _| Inst::IntBin {
                    ty,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    Ok(match ty {
                        IntTy::I32 => SpecVal::I32(bin_i32(op, as_i32(x[0]), as_i32(x[1]))?),
                        IntTy::I64 => SpecVal::I64(bin_i64(op, as_i64(x[0]), as_i64(x[1]))?),
                    })
                }),
            });
        }

        for op in CmpOp::ALL {
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt, vt],
                result: ValType::I32,
                class: Class::Pure,
                shape: Shape::Operands,
                encoding: enc_int_cmp(ty, op),
                build: Box::new(move |v, _| Inst::IntCmp {
                    ty,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    Ok(SpecVal::I32(match ty {
                        IntTy::I32 => cmp_i32(op, as_i32(x[0]), as_i32(x[1])),
                        IntTy::I64 => cmp_i64(op, as_i64(x[0]), as_i64(x[1])),
                    }))
                }),
            });
        }

        for op in IntUnOp::ALL {
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt],
                result: vt,
                class: Class::Pure,
                shape: Shape::Operands,
                encoding: enc_int_un(ty, op),
                build: Box::new(move |v, _| Inst::IntUn { ty, op, a: v[0] }),
                eval: Box::new(move |x| {
                    Ok(match ty {
                        IntTy::I32 => SpecVal::I32(un_i32(op, as_i32(x[0]))),
                        IntTy::I64 => SpecVal::I64(un_i64(op, as_i64(x[0]))),
                    })
                }),
            });
        }

        push(OpRow {
            id: format!("{}.eqz", ty.prefix()),
            operands: vec![vt],
            result: ValType::I32,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(match ty {
                IntTy::I32 => 0x30,
                IntTy::I64 => 0x50,
            }),
            build: Box::new(move |v, _| Inst::Eqz { ty, a: v[0] }),
            eval: Box::new(move |x| {
                let zero = match ty {
                    IntTy::I32 => as_i32(x[0]) == 0,
                    IntTy::I64 => as_i64(x[0]) == 0,
                };
                Ok(SpecVal::I32(zero as i32))
            }),
        });

        // `select` is polymorphic (the verifier's one polymorphic op); slice 1
        // instantiates the integer types.
        push(OpRow {
            id: format!("select ({})", ty.prefix()),
            operands: vec![ValType::I32, vt, vt],
            result: vt,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(0x70),
            build: Box::new(|v, _| Inst::Select {
                cond: v[0],
                a: v[1],
                b: v[2],
            }),
            eval: Box::new(|x| Ok(if as_i32(x[0]) != 0 { x[1] } else { x[2] })),
        });
    }

    for op in [ConvOp::ExtendI32S, ConvOp::ExtendI32U, ConvOp::WrapI64] {
        let (name, from, to) = op.sig();
        push(OpRow {
            id: name.into(),
            operands: vec![from],
            result: to,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: enc_convert(op),
            build: Box::new(move |v, _| Inst::Convert { op, a: v[0] }),
            eval: Box::new(move |x| {
                Ok(match op {
                    ConvOp::ExtendI32S => SpecVal::I64(as_i32(x[0]) as i64),
                    ConvOp::ExtendI32U => SpecVal::I64(as_i32(x[0]) as u32 as i64),
                    ConvOp::WrapI64 => SpecVal::I32(as_i64(x[0]) as i32),
                })
            }),
        });
    }

    for op in CastOp::ALL {
        let (name, from, to) = op.sig();
        push(OpRow {
            id: name.into(),
            operands: vec![from],
            result: to,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: enc_cast(op),
            build: Box::new(move |v, _| Inst::Cast { op, a: v[0] }),
            eval: Box::new(move |x| Ok(cast(op, x[0]))),
        });
    }

    // Pointer ops (§3b/§10): plain `i64` arithmetic off-CHERI — `ptr.add` wraps,
    // the casts pass the value through.
    push(OpRow {
        id: "ptr.add".into(),
        operands: vec![ValType::I64, ValType::I64],
        result: ValType::I64,
        class: Class::Pure,
        shape: Shape::Operands,
        encoding: Enc::Byte(0x78),
        build: Box::new(|v, _| Inst::PtrAdd { a: v[0], b: v[1] }),
        eval: Box::new(|x| Ok(SpecVal::I64(as_i64(x[0]).wrapping_add(as_i64(x[1]))))),
    });
    for to_int in [false, true] {
        push(OpRow {
            id: format!("ptr.{}", if to_int { "to_int" } else { "from_int" }),
            operands: vec![ValType::I64],
            result: ValType::I64,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(if to_int { 0x77 } else { 0x76 }),
            build: Box::new(move |v, _| Inst::PtrCast { to_int, a: v[0] }),
            eval: Box::new(|x| Ok(x[0])),
        });
    }

    rows
}

/// Slice-2 rows: scalar floats — consts, `FBin`/`FUn`/`Fma`/`FCmp`, the saturating and
/// trapping float→int conversions, int→float, and the float `select` instantiations.
pub fn float_rows() -> Vec<OpRow> {
    let mut rows: Vec<OpRow> = Vec::new();
    let mut push = |row: OpRow| rows.push(row);

    push(OpRow {
        id: "f32.const".into(),
        operands: vec![],
        result: ValType::F32,
        class: Class::Pure,
        shape: Shape::Immediate,
        encoding: Enc::Byte(0x12),
        build: Box::new(|_, imm| match imm[0] {
            SpecVal::F32(b) => Inst::ConstF32(b),
            _ => unreachable!("spec row fed a non-f32 immediate"),
        }),
        eval: Box::new(|x| Ok(x[0])),
    });
    push(OpRow {
        id: "f64.const".into(),
        operands: vec![],
        result: ValType::F64,
        class: Class::Pure,
        shape: Shape::Immediate,
        encoding: Enc::Byte(0x13),
        build: Box::new(|_, imm| match imm[0] {
            SpecVal::F64(b) => Inst::ConstF64(b),
            _ => unreachable!("spec row fed a non-f64 immediate"),
        }),
        eval: Box::new(|x| Ok(x[0])),
    });

    for ty in [FloatTy::F32, FloatTy::F64] {
        let vt = ty.val();

        for op in FBinOp::ALL {
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt, vt],
                result: vt,
                class: Class::Pure,
                shape: Shape::Operands,
                encoding: enc_fbin(ty, op),
                build: Box::new(move |v, _| Inst::FBin {
                    ty,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    Ok(match ty {
                        FloatTy::F32 => {
                            SpecVal::F32(fbin_f32(op, as_f32(x[0]), as_f32(x[1])).to_bits())
                        }
                        FloatTy::F64 => {
                            SpecVal::F64(fbin_f64(op, as_f64(x[0]), as_f64(x[1])).to_bits())
                        }
                    })
                }),
            });
        }

        for op in FUnOp::ALL {
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt],
                result: vt,
                class: Class::Pure,
                shape: Shape::Operands,
                encoding: enc_fun(ty, op),
                build: Box::new(move |v, _| Inst::FUn { ty, op, a: v[0] }),
                eval: Box::new(move |x| {
                    Ok(match ty {
                        FloatTy::F32 => SpecVal::F32(fun_f32(op, as_f32(x[0])).to_bits()),
                        FloatTy::F64 => SpecVal::F64(fun_f64(op, as_f64(x[0])).to_bits()),
                    })
                }),
            });
        }

        // Fused multiply-add: a·b + c with a single rounding (§3b — the correctly-
        // rounded FMA, not mul-then-add).
        push(OpRow {
            id: format!("{}.fma", ty.prefix()),
            operands: vec![vt, vt, vt],
            result: vt,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(0x7D),
            build: Box::new(move |v, _| Inst::Fma {
                ty,
                a: v[0],
                b: v[1],
                c: v[2],
            }),
            eval: Box::new(move |x| {
                Ok(match ty {
                    FloatTy::F32 => {
                        SpecVal::F32(as_f32(x[0]).mul_add(as_f32(x[1]), as_f32(x[2])).to_bits())
                    }
                    FloatTy::F64 => {
                        SpecVal::F64(as_f64(x[0]).mul_add(as_f64(x[1]), as_f64(x[2])).to_bits())
                    }
                })
            }),
        });

        for op in FCmpOp::ALL {
            push(OpRow {
                id: format!("{}.{}", ty.prefix(), op.name()),
                operands: vec![vt, vt],
                result: ValType::I32,
                class: Class::Pure,
                shape: Shape::Operands,
                encoding: enc_fcmp(ty, op),
                build: Box::new(move |v, _| Inst::FCmp {
                    ty,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    // Promoting f32 → f64 is exact and order-preserving, so one f64
                    // comparison covers both widths.
                    let (a, b) = match ty {
                        FloatTy::F32 => (as_f32(x[0]) as f64, as_f32(x[1]) as f64),
                        FloatTy::F64 => (as_f64(x[0]), as_f64(x[1])),
                    };
                    Ok(SpecVal::I32(fcmp(op, a, b)))
                }),
            });
        }

        push(OpRow {
            id: format!("select ({})", ty.prefix()),
            operands: vec![ValType::I32, vt, vt],
            result: vt,
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(0x70),
            build: Box::new(|v, _| Inst::Select {
                cond: v[0],
                a: v[1],
                b: v[2],
            }),
            eval: Box::new(|x| Ok(if as_i32(x[0]) != 0 { x[1] } else { x[2] })),
        });
    }

    for op in FToI::ALL {
        let (from, to, _) = op.parts();
        push(OpRow {
            id: op.name().into(),
            operands: vec![from.val()],
            result: to.val(),
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: Enc::Byte(0xD0 + ftoi_off(op)),
            build: Box::new(move |v, _| Inst::FToISat { op, a: v[0] }),
            eval: Box::new(move |x| Ok(trunc_sat(op, x[0]))),
        });
        push(OpRow {
            id: op.trap_name().into(),
            operands: vec![from.val()],
            result: to.val(),
            class: Class::Trapping,
            shape: Shape::Operands,
            encoding: Enc::Byte(0xD8 + ftoi_off(op)),
            build: Box::new(move |v, _| Inst::FToITrap { op, a: v[0] }),
            eval: Box::new(move |x| trunc_trap(op, x[0])),
        });
    }

    for op in IToF::ALL {
        let (from, to, _) = op.parts();
        push(OpRow {
            id: op.name().into(),
            operands: vec![from.val()],
            result: to.val(),
            class: Class::Pure,
            shape: Shape::Operands,
            encoding: enc_itof(op),
            build: Box::new(move |v, _| Inst::IToFConv { op, a: v[0] }),
            eval: Box::new(move |x| Ok(itof(op, x[0]))),
        });
    }

    rows
}

/// Every specced row (grows slice by slice until [`coverage`] is fully realized).
pub fn all_rows() -> Vec<OpRow> {
    let mut rows = scalar_rows();
    rows.extend(float_rows());
    rows
}

// --- input vectors -------------------------------------------------------------------

/// Boundary-biased input pools (SPEC.md suite 1). Every unary/binary op takes the full
/// cross product of its pools; only wider arities are strided down to [`VECTOR_CAP`].
pub const I32_INPUTS: &[i32] = &[
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    i32::MIN,
    i32::MIN + 1,
    i32::MAX,
    i32::MAX - 1,
    0x7f,
    0x80,
    0xff,
    0x100,
    0x7fff,
    0x8000,
    0xffff,
    0x10000,
    31,
    32,
    33,
    -31,
    -32,
    0x0102_0304,
    0x5555_5555,
    -0x5555_5556,
    0x4000_0000,
    -0x4000_0000,
];

pub const I64_INPUTS: &[i64] = &[
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    i64::MIN,
    i64::MIN + 1,
    i64::MAX,
    i64::MAX - 1,
    0x7f,
    0x80,
    0xff,
    0x7fff,
    0x8000,
    0xffff,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x1_0000_0000,
    i32::MIN as i64,
    i32::MIN as i64 - 1,
    i32::MAX as i64 + 1,
    31,
    32,
    33,
    63,
    64,
    65,
    -63,
    0x0102_0304_0506_0708,
    0x5555_5555_5555_5555,
    -0x5555_5555_5555_5556,
];

/// `f32` inputs as raw bits: signed zeros/ones, halves, sub/normal boundaries, max
/// finite, infinities, quiet + signalling NaNs, and powers of two near the int ranges.
pub const F32_INPUTS: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x3fc0_0000, // 1.5
    0xc020_0000, // -2.5
    0x0000_0001, // min positive subnormal
    0x8000_0001, // min negative subnormal
    0x007f_ffff, // max subnormal
    0x0080_0000, // min positive normal
    0x7f7f_ffff, // max finite
    0xff7f_ffff, // min finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // canonical qNaN
    0xffc0_0001, // negative qNaN with payload
    0x7f80_0001, // sNaN
    0x4b00_0000, // 2^23 (integer-precision boundary)
    // Float→int trap/saturation boundaries (§3b `trunc` vs `trunc_sat`): the largest
    // representable f32 on the fitting side of each bound, the exact bound, and one
    // step past it.
    0x4eff_ffff, // 2147483520 — largest f32 < 2^31 (fits i32 signed)
    0x4f00_0000, // 2^31 (traps i32 signed)
    0xcf00_0000, // -2^31 exactly (fits i32 signed)
    0xcf00_0001, // one f32 below -2^31 (traps)
    0x4f7f_ffff, // largest f32 < 2^32 (fits u32)
    0x4f80_0000, // 2^32 (traps u32)
    0x5eff_ffff, // largest f32 < 2^63 (fits i64 signed)
    0x5f00_0000, // 2^63 (traps i64 signed; fits u64)
    0xdf00_0000, // -2^63 exactly (fits i64 signed)
    0xdf00_0001, // one f32 below -2^63 (traps)
    0x5f7f_ffff, // largest f32 < 2^64 (fits u64)
    0x5f80_0000, // 2^64 (traps u64)
    0xbf00_0000, // -0.5 (truncation -0 fits every unsigned target)
    0xbf7f_ffff, // just above -1.0 (still fits unsigned)
    0x4049_0fdb, // π
];

/// `f64` inputs as raw bits — same families as [`F32_INPUTS`], plus demote-rounding
/// halfway cases (ties-to-even at the f32 precision boundary) and an overflow-to-inf.
pub const F64_INPUTS: &[u64] = &[
    0x0000_0000_0000_0000, // +0
    0x8000_0000_0000_0000, // -0
    0x3ff0_0000_0000_0000, // 1.0
    0xbff0_0000_0000_0000, // -1.0
    0x3ff8_0000_0000_0000, // 1.5
    0xc004_0000_0000_0000, // -2.5
    0x0000_0000_0000_0001, // min positive subnormal
    0x8000_0000_0000_0001,
    0x000f_ffff_ffff_ffff, // max subnormal
    0x0010_0000_0000_0000, // min positive normal
    0x7fef_ffff_ffff_ffff, // max finite
    0xffef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000, // +inf
    0xfff0_0000_0000_0000, // -inf
    0x7ff8_0000_0000_0000, // canonical qNaN
    0xfff8_0000_0000_0001, // negative qNaN with payload
    0x7ff0_0000_0000_0001, // sNaN
    0x3ff0_0000_1000_0000, // 1 + 2^-24: demote ties-to-even → 1.0
    0x3ff0_0000_3000_0000, // 1 + 3·2^-24: demote ties-to-even → 1 + 2^-22
    0x47f0_0000_0000_0000, // 2^128: demote overflows → +inf
    0x4340_0000_0000_0000, // 2^53 (integer-precision boundary)
    // Float→int trap/saturation boundaries, as in `F32_INPUTS` but at f64 precision
    // (2^31−1 and fractional neighbors of the bounds are exact here).
    0x41df_ffff_ffc0_0000, // 2147483647.0 = i32::MAX exactly
    0x41df_ffff_ffe0_0000, // 2147483647.5 (truncates to i32::MAX — fits signed)
    0x41e0_0000_0000_0000, // 2147483648.0 = 2^31 (traps i32 signed)
    0xc1e0_0000_0000_0000, // -2^31 exactly (fits i32 signed)
    0xc1e0_0000_0010_0000, // -2147483648.5 (truncates to i32::MIN — fits)
    0xc1e0_0000_0020_0000, // -2147483649.0 (traps i32 signed)
    0x41ef_ffff_ffff_ffff, // 4294967295.999… (truncates to u32::MAX — fits unsigned)
    0x41f0_0000_0000_0000, // 2^32 (traps u32)
    0x43df_ffff_ffff_ffff, // largest f64 < 2^63 (fits i64 signed)
    0x43e0_0000_0000_0000, // 2^63 (traps i64 signed; fits u64)
    0xc3e0_0000_0000_0000, // -2^63 exactly (fits i64 signed)
    0xc3e0_0000_0000_0001, // one f64 below -2^63 (traps)
    0x43ef_ffff_ffff_ffff, // largest f64 < 2^64 (fits u64)
    0x43f0_0000_0000_0000, // 2^64 (traps u64)
    0xbfe0_0000_0000_0000, // -0.5 (fits every unsigned target)
    0xbfef_ffff_ffff_ffff, // just above -1.0 (still fits unsigned)
    0x4009_21fb_5444_2d18, // π
];

/// Deterministic cap on a row's vector count. Chosen so every unary and binary pool
/// cross product fits **whole** (the boundary lattice is never strided away); only
/// 3-operand `select` gets strided. If a row is strided, the suite still covers the
/// pools evenly (fixed stride over the mixed-radix enumeration) — no randomness.
pub const VECTOR_CAP: usize = 2048;

fn pool(t: ValType) -> Vec<SpecVal> {
    match t {
        ValType::I32 => I32_INPUTS.iter().map(|&x| SpecVal::I32(x)).collect(),
        ValType::I64 => I64_INPUTS.iter().map(|&x| SpecVal::I64(x)).collect(),
        ValType::F32 => F32_INPUTS.iter().map(|&x| SpecVal::F32(x)).collect(),
        ValType::F64 => F64_INPUTS.iter().map(|&x| SpecVal::F64(x)).collect(),
        ValType::V128 | ValType::Ref => unreachable!("no slice-1 row takes {t:?}"),
    }
}

/// The row's input vectors: the boundary-pool cross product, strided down to
/// [`VECTOR_CAP`] when wider arities explode (deterministic; see [`VECTOR_CAP`]).
pub fn vectors_for(row: &OpRow) -> Vec<Vec<SpecVal>> {
    let pools: Vec<Vec<SpecVal>> = row.inputs().into_iter().map(pool).collect();
    let total: usize = pools.iter().map(|p| p.len()).product();
    let stride = total.div_ceil(VECTOR_CAP).max(1);
    (0..total)
        .step_by(stride)
        .map(|i| {
            let mut rest = i;
            pools
                .iter()
                .map(|p| {
                    let v = p[rest % p.len()];
                    rest /= p.len();
                    v
                })
                .collect()
        })
        .collect()
}

// --- module construction --------------------------------------------------------------

/// The single-op module for a row: `func(operands) -> result { r = op(params); return r }`
/// (an `Immediate` row bakes `vector[0]` and takes no params). The result is *returned*,
/// so a trapping op is never dead code a backend may elide.
pub fn module_for(row: &OpRow, vector: &[SpecVal]) -> Module {
    let params = match row.shape {
        Shape::Operands => row.operands.clone(),
        Shape::Immediate => Vec::new(),
    };
    let idx: Vec<ValIdx> = (0..params.len() as ValIdx).collect();
    let inst = (row.build)(&idx, vector);
    let ridx = params.len() as ValIdx;
    Module {
        funcs: vec![Func {
            params: params.clone(),
            results: vec![row.result],
            blocks: vec![Block {
                params,
                insts: vec![inst],
                term: Terminator::Return(vec![ridx]),
            }],
        }],
        ..Default::default()
    }
}

// --- the memory window model (SPEC.md slice 5) ----------------------------------------
//
// Trap-confinement semantics per DESIGN.md §4 / TRAP_CONFINEMENT.md / the `svm-mask`
// contract, restated independently: a `width`-byte access at `addr` with immediate
// `offset` is admitted iff the whole span `[addr+offset, addr+offset+width)` — computed
// **without wraparound** — lies within `[0, mapped)`; anything else (including a
// wrapping effective address) raises `MemoryFault` at the access. No masking, no
// aliasing. Memory is little-endian (§3b). The spec window is fully mapped
// (`mapped == reserved`); the decoupled reserved-tail model rides the existing
// escape-oracle differential.

/// The spec window: 4 KiB, fully mapped.
pub const MEM_LOG2: u8 = 12;
pub const MEM_SIZE: u64 = 1 << MEM_LOG2;

/// Admit a scalar access, returning the effective address as a window index.
fn admit(addr: u64, offset: u64, width: u32, mapped: u64) -> Result<usize, SpecTrap> {
    match addr
        .checked_add(offset)
        .and_then(|ea| ea.checked_add(width as u64))
    {
        Some(end) if end <= mapped => Ok((end - width as u64) as usize),
        _ => Err(SpecTrap::MemoryFault),
    }
}

/// Admit a bulk span `[ptr, ptr+len)` (D62): checked as a whole, overflow-free; a
/// zero-length op is a no-op even at a wild pointer.
fn admit_span(ptr: u64, len: u64, mapped: u64) -> Result<(), SpecTrap> {
    if len == 0 {
        return Ok(());
    }
    match ptr.checked_add(len) {
        Some(end) if end <= mapped => Ok(()),
        _ => Err(SpecTrap::MemoryFault),
    }
}

/// The little-endian bit pattern a store writes (the value's low `width` bytes).
fn store_bits(v: SpecVal) -> u64 {
    match v {
        SpecVal::I32(x) => x as u32 as u64,
        SpecVal::I64(x) => x as u64,
        SpecVal::F32(b) => b as u64,
        SpecVal::F64(b) => b,
    }
}

fn load_eval(op: LoadOp, addr: u64, offset: u64, win: &[u8]) -> Result<SpecVal, SpecTrap> {
    let (_, _, width, _) = op.info();
    let ea = admit(addr, offset, width, win.len() as u64)?;
    let mut raw = [0u8; 8];
    raw[..width as usize].copy_from_slice(&win[ea..ea + width as usize]);
    let u = u64::from_le_bytes(raw);
    Ok(match op {
        LoadOp::I32 => SpecVal::I32(u as u32 as i32),
        LoadOp::I64 => SpecVal::I64(u as i64),
        LoadOp::F32 => SpecVal::F32(u as u32),
        LoadOp::F64 => SpecVal::F64(u),
        LoadOp::I32_8S => SpecVal::I32(u as u8 as i8 as i32),
        LoadOp::I32_8U => SpecVal::I32(u as u8 as i32),
        LoadOp::I32_16S => SpecVal::I32(u as u16 as i16 as i32),
        LoadOp::I32_16U => SpecVal::I32(u as u16 as i32),
        LoadOp::I64_8S => SpecVal::I64(u as u8 as i8 as i64),
        LoadOp::I64_8U => SpecVal::I64(u as u8 as i64),
        LoadOp::I64_16S => SpecVal::I64(u as u16 as i16 as i64),
        LoadOp::I64_16U => SpecVal::I64(u as u16 as i64),
        LoadOp::I64_32S => SpecVal::I64(u as u32 as i32 as i64),
        LoadOp::I64_32U => SpecVal::I64(u as u32 as i64),
    })
}

fn store_eval(
    op: StoreOp,
    addr: u64,
    offset: u64,
    value: SpecVal,
    win: &mut [u8],
) -> Result<(), SpecTrap> {
    let (_, _, width) = op.info();
    let ea = admit(addr, offset, width, win.len() as u64)?;
    let bytes = store_bits(value).to_le_bytes();
    win[ea..ea + width as usize].copy_from_slice(&bytes[..width as usize]);
    Ok(())
}

/// A memory-op row: like [`OpRow`] but observed through the **final window** (plus a
/// result for loads), evaluated against the spec window model. `Load`/`Store` carry an
/// immediate `offset`, so each `(row, offset)` pair is its own module.
pub struct MemRow {
    pub id: String,
    /// SSA operand types, fed as function parameters (addresses/lengths are `i64`).
    pub operands: Vec<ValType>,
    /// The loaded value's type; `None` for stores and bulk ops.
    pub result: Option<ValType>,
    pub encoding: Enc,
    /// Per-operand input pools, aligned with `operands` (addresses and lengths get
    /// window-boundary pools, store values get bit-pattern pools).
    pub pools: Vec<Vec<SpecVal>>,
    /// Whether the op takes an immediate offset (`Load`/`Store`; bulk ops don't).
    pub has_offset: bool,
    /// Construct the op over operand value indices plus the immediate offset.
    #[allow(clippy::type_complexity)]
    pub build: Box<dyn Fn(&[ValIdx], u64) -> Inst>,
    /// Evaluate against the model window: mutates `win`, returns the loaded value (if
    /// any) or the trap. A trapping access mutates nothing (checked before access).
    #[allow(clippy::type_complexity)]
    pub eval: Box<dyn Fn(&[SpecVal], u64, &mut [u8]) -> Result<Option<SpecVal>, SpecTrap>>,
}

/// Addresses (as `i64` params) biased to the window-boundary lattice: in-window,
/// exactly-fitting, one-past, far-past, and wrap-around values.
fn addr_pool() -> Vec<SpecVal> {
    [
        0,
        1,
        2,
        7,
        8,
        64,
        0xabc,
        MEM_SIZE - 8,
        MEM_SIZE - 4,
        MEM_SIZE - 2,
        MEM_SIZE - 1,
        MEM_SIZE,
        MEM_SIZE + 1,
        u64::MAX,     // wrap-around: must fault, never alias
        u64::MAX - 6, // wraps for width 8, faults for width ≤ 6 by span, too
        1 << 63,
        (1 << 63) - 4,
    ]
    .into_iter()
    .map(|a| SpecVal::I64(a as i64))
    .collect()
}

/// Bulk-op lengths: zero (a no-op even at a wild pointer, D62), small, page-sized,
/// one-past, and overflowing.
fn len_pool() -> Vec<SpecVal> {
    [0, 1, 5, 16, 256, MEM_SIZE, MEM_SIZE + 1, u64::MAX]
        .into_iter()
        .map(|l| SpecVal::I64(l as i64))
        .collect()
}

/// Store-value bit patterns per type (moves only — no arithmetic — so these pin
/// bit-exact round-trips through the window, NaNs included).
fn store_val_pool(t: ValType) -> Vec<SpecVal> {
    match t {
        ValType::I32 => [0, -1, 0x0102_0304, 0x80]
            .into_iter()
            .map(SpecVal::I32)
            .collect(),
        ValType::I64 => [0, -1, 0x0102_0304_0506_0708, 0x80]
            .into_iter()
            .map(SpecVal::I64)
            .collect(),
        ValType::F32 => [0x3f80_0000, 0x7fc0_0000, 0x8000_0001]
            .into_iter()
            .map(SpecVal::F32)
            .collect(),
        ValType::F64 => [
            0x3ff0_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x8000_0000_0000_0001,
        ]
        .into_iter()
        .map(SpecVal::F64)
        .collect(),
        ValType::V128 | ValType::Ref => unreachable!("no slice-5 store of {t:?}"),
    }
}

fn as_u64(v: SpecVal) -> u64 {
    as_i64(v) as u64
}

/// The slice-5 rows: the 14 loads, 9 stores, and the 3 bulk ops.
pub fn mem_rows() -> Vec<MemRow> {
    let mut rows: Vec<MemRow> = Vec::new();

    for (i, op) in LoadOp::ALL.into_iter().enumerate() {
        let (name, result, _, _) = op.info();
        rows.push(MemRow {
            id: name.into(),
            operands: vec![ValType::I64],
            result: Some(result),
            encoding: Enc::Byte(0xF0 + i as u8),
            pools: vec![addr_pool()],
            has_offset: true,
            build: Box::new(move |v, off| Inst::Load {
                op,
                addr: v[0],
                offset: off,
                align: 0,
            }),
            eval: Box::new(move |x, off, win| load_eval(op, as_u64(x[0]), off, win).map(Some)),
        });
    }

    for (i, op) in StoreOp::ALL.into_iter().enumerate() {
        let (name, value_ty, _) = op.info();
        rows.push(MemRow {
            id: name.into(),
            operands: vec![ValType::I64, value_ty],
            result: None,
            encoding: Enc::Byte(0x84 + i as u8),
            pools: vec![addr_pool(), store_val_pool(value_ty)],
            has_offset: true,
            build: Box::new(move |v, off| Inst::Store {
                op,
                addr: v[0],
                value: v[1],
                offset: off,
                align: 0,
            }),
            eval: Box::new(move |x, off, win| {
                store_eval(op, as_u64(x[0]), off, x[1], win).map(|()| None)
            }),
        });
    }

    // mem.copy — defined for **non-overlapping** spans (§3b/D62; the vector generator
    // filters overlap out, see `mem_vectors_for`). Both spans admitted as a whole
    // before any byte moves.
    rows.push(MemRow {
        id: "mem.copy".into(),
        operands: vec![ValType::I64; 3],
        result: None,
        encoding: Enc::Byte(0x8D),
        pools: vec![addr_pool(), addr_pool(), len_pool()],
        has_offset: false,
        build: Box::new(|v, _| Inst::MemCopy {
            dst: v[0],
            src: v[1],
            len: v[2],
        }),
        eval: Box::new(|x, _, win| {
            let (dst, src, len) = (as_u64(x[0]), as_u64(x[1]), as_u64(x[2]));
            admit_span(dst, len, win.len() as u64)?;
            admit_span(src, len, win.len() as u64)?;
            if len == 0 {
                return Ok(None); // a no-op even at a wild pointer (D62)
            }
            let snap = win[src as usize..(src + len) as usize].to_vec();
            win[dst as usize..(dst + len) as usize].copy_from_slice(&snap);
            Ok(None)
        }),
    });
    // mem.move — overlap-safe (as-if through a snapshot).
    rows.push(MemRow {
        id: "mem.move".into(),
        operands: vec![ValType::I64; 3],
        result: None,
        encoding: Enc::Byte(0x8E),
        pools: vec![addr_pool(), addr_pool(), len_pool()],
        has_offset: false,
        build: Box::new(|v, _| Inst::MemMove {
            dst: v[0],
            src: v[1],
            len: v[2],
        }),
        eval: Box::new(|x, _, win| {
            let (dst, src, len) = (as_u64(x[0]), as_u64(x[1]), as_u64(x[2]));
            admit_span(dst, len, win.len() as u64)?;
            admit_span(src, len, win.len() as u64)?;
            if len == 0 {
                return Ok(None); // a no-op even at a wild pointer (D62)
            }
            let snap = win[src as usize..(src + len) as usize].to_vec();
            win[dst as usize..(dst + len) as usize].copy_from_slice(&snap);
            Ok(None)
        }),
    });
    // mem.fill — writes the fill value's low byte across the span.
    rows.push(MemRow {
        id: "mem.fill".into(),
        operands: vec![ValType::I64, ValType::I32, ValType::I64],
        result: None,
        encoding: Enc::Byte(0x8F),
        pools: vec![addr_pool(), store_val_pool(ValType::I32), len_pool()],
        has_offset: false,
        build: Box::new(|v, _| Inst::MemFill {
            dst: v[0],
            val: v[1],
            len: v[2],
        }),
        eval: Box::new(|x, _, win| {
            let (dst, val, len) = (as_u64(x[0]), as_i32(x[1]), as_u64(x[2]));
            admit_span(dst, len, win.len() as u64)?;
            if len == 0 {
                return Ok(None); // a no-op even at a wild pointer (D62)
            }
            win[dst as usize..(dst + len) as usize].fill(val as u8);
            Ok(None)
        }),
    });

    rows
}

/// Load/store immediate-offset pool: each distinct offset is its own module (the
/// offset is an instruction immediate, not an operand).
pub const MEM_OFFSETS: &[u64] = &[0, 1, 8, MEM_SIZE - 8, MEM_SIZE, u64::MAX];

/// Deterministic strided cross product of the row's operand pools (shared shape with
/// [`vectors_for`]). `mem.copy` vectors with genuinely-overlapping spans are filtered
/// out — the IR defines `mem.copy` for non-overlapping spans only (its overlap
/// behavior is deliberately unpinned; `mem.move` is the overlap-safe op).
pub fn mem_vectors_for(row: &MemRow) -> Vec<Vec<SpecVal>> {
    let total: usize = row.pools.iter().map(|p| p.len()).product();
    let stride = total.div_ceil(VECTOR_CAP).max(1);
    (0..total)
        .step_by(stride)
        .map(|i| {
            let mut rest = i;
            row.pools
                .iter()
                .map(|p| {
                    let v = p[rest % p.len()];
                    rest /= p.len();
                    v
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| {
            if row.id != "mem.copy" {
                return true;
            }
            // Keep only vectors where the admitted spans cannot overlap: either the
            // copy faults (span check fails — still a valid vector) or dst/src are
            // disjoint.
            let (dst, src, len) = (as_u64(v[0]), as_u64(v[1]), as_u64(v[2]));
            let faults =
                admit_span(dst, len, MEM_SIZE).is_err() || admit_span(src, len, MEM_SIZE).is_err();
            faults || len == 0 || dst + len <= src || src + len <= dst
        })
        .collect()
}

/// The single-op module for a memory row at one immediate `offset`: declares the 4 KiB
/// spec window, takes the operands as params, returns the loaded value (if any).
pub fn module_for_mem(row: &MemRow, offset: u64) -> Module {
    let params = row.operands.clone();
    let idx: Vec<ValIdx> = (0..params.len() as ValIdx).collect();
    let inst = (row.build)(&idx, offset);
    let term = match row.result {
        Some(_) => Terminator::Return(vec![params.len() as ValIdx]),
        None => Terminator::Return(vec![]),
    };
    Module {
        funcs: vec![Func {
            params: params.clone(),
            results: row.result.into_iter().collect(),
            blocks: vec![Block {
                params,
                insts: vec![inst],
                term,
            }],
        }],
        memory: Some(Memory {
            size_log2: MEM_LOG2,
        }),
        ..Default::default()
    }
}

// --- coverage: the compile-time completeness hook -------------------------------------

/// Classify every instruction — **exhaustively, no wildcard** (SPEC.md "drift
/// protection"): adding an `Inst` variant fails this crate's build until the spec makes
/// a conscious decision about it. Classifications for ops whose slice hasn't landed yet
/// are provisional (they gate nothing until their rows/suites exist).
pub fn coverage(inst: &Inst) -> Class {
    match inst {
        // Slice 1 — the scalar deterministic core.
        Inst::ConstI32(..) | Inst::ConstI64(..) => Class::Pure,
        Inst::IntBin { op, .. } => match op {
            BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => Class::Trapping,
            _ => Class::Pure,
        },
        Inst::IntCmp { .. } | Inst::IntUn { .. } | Inst::Eqz { .. } => Class::Pure,
        Inst::Convert { .. } | Inst::Select { .. } => Class::Pure,
        Inst::Cast { .. } => Class::Pure,
        Inst::PtrAdd { .. } | Inst::PtrCast { .. } => Class::Pure,

        // Slice 2 — scalar floats.
        Inst::ConstF32(..) | Inst::ConstF64(..) => Class::Pure,
        Inst::FBin { .. } | Inst::FUn { .. } | Inst::Fma { .. } | Inst::FCmp { .. } => Class::Pure,
        Inst::FToISat { .. } | Inst::IToFConv { .. } => Class::Pure,
        Inst::FToITrap { .. } => Class::Trapping,

        // Slice 5 — the memory window model.
        Inst::Load { .. } | Inst::Store { .. } => Class::Memory,
        Inst::MemCopy { .. } | Inst::MemMove { .. } | Inst::MemFill { .. } => Class::Memory,
        Inst::AtomicLoad { .. }
        | Inst::AtomicStore { .. }
        | Inst::AtomicRmw { .. }
        | Inst::AtomicCmpxchg { .. }
        | Inst::AtomicFence { .. } => Class::Memory,
        Inst::V128Load { .. } | Inst::V128Store { .. } => Class::Memory,

        // Slice 6 — SIMD value ops.
        Inst::ConstV128(..)
        | Inst::Splat { .. }
        | Inst::ExtractLane { .. }
        | Inst::ReplaceLane { .. }
        | Inst::VIntBin { .. }
        | Inst::VIntCmp { .. }
        | Inst::VFloatCmp { .. }
        | Inst::VShift { .. }
        | Inst::VIntUn { .. }
        | Inst::VSatBin { .. }
        | Inst::VWiden { .. }
        | Inst::VNarrow { .. }
        | Inst::VConvert { .. }
        | Inst::VPMinMax { .. }
        | Inst::VPopcnt { .. }
        | Inst::VAvgr { .. }
        | Inst::VDot { .. }
        | Inst::VDotI8 { .. }
        | Inst::VExtMul { .. }
        | Inst::VExtAddPairwise { .. }
        | Inst::VQ15MulrSat { .. }
        | Inst::VFma { .. }
        | Inst::VAnyTrue { .. }
        | Inst::VAllTrue { .. }
        | Inst::VBitmask { .. }
        | Inst::VFloatBin { .. }
        | Inst::VFloatUn { .. }
        | Inst::VBitBin { .. }
        | Inst::VNot { .. }
        | Inst::Bitselect { .. }
        | Inst::Shuffle { .. }
        | Inst::Swizzle { .. }
        | Inst::SimdWidthBytes => Class::Pure,

        // Slice 7 — typing/encoding rows only (semantics out of scope, SPEC.md).
        Inst::Call { .. } | Inst::CallIndirect { .. } | Inst::RefFunc { .. } => Class::Control,
        Inst::SetJmp { .. } | Inst::LongJmp { .. } | Inst::GcRoots { .. } => Class::Control,
        Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } | Inst::DurableShadowBase => Class::Control,
        Inst::CapCall { .. }
        | Inst::CallImport { .. }
        | Inst::CapSelfCount
        | Inst::CapSelfGet { .. }
        | Inst::CapSelfResolve { .. }
        | Inst::CapSelfLabel { .. } => Class::Host,
        Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. } => Class::Concurrency,
    }
}

/// Terminator counterpart of [`coverage`] — same exhaustive-match forcing function.
pub fn coverage_term(term: &Terminator) -> Class {
    match term {
        Terminator::Br { .. }
        | Terminator::BrIf { .. }
        | Terminator::BrTable { .. }
        | Terminator::Return(..)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => Class::Control,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §3b trap lattice, pinned as spec-internal expectations (these are the rows'
    /// *definitions* — if one of these fails, the spec itself regressed).
    #[test]
    fn division_trap_lattice() {
        assert_eq!(bin_i32(BinOp::DivS, 7, 0), Err(SpecTrap::DivByZero));
        assert_eq!(bin_i32(BinOp::DivU, 7, 0), Err(SpecTrap::DivByZero));
        assert_eq!(bin_i32(BinOp::RemS, 7, 0), Err(SpecTrap::DivByZero));
        assert_eq!(bin_i32(BinOp::RemU, 7, 0), Err(SpecTrap::DivByZero));
        assert_eq!(
            bin_i32(BinOp::DivS, i32::MIN, -1),
            Err(SpecTrap::IntOverflow)
        );
        // Only the *quotient* overflows on INT_MIN/−1 — the remainder 0 is
        // representable, so rem_s returns it (§3b; wasm-identical).
        assert_eq!(bin_i32(BinOp::RemS, i32::MIN, -1), Ok(0));
        assert_eq!(
            bin_i64(BinOp::DivS, i64::MIN, -1),
            Err(SpecTrap::IntOverflow)
        );
        assert_eq!(bin_i64(BinOp::RemS, i64::MIN, -1), Ok(0));
        // Unsigned division cannot overflow.
        assert_eq!(bin_i32(BinOp::DivU, i32::MIN, -1), Ok(0));
    }

    #[test]
    fn shift_amounts_mod_bitwidth() {
        assert_eq!(bin_i32(BinOp::Shl, 1, 32), Ok(1));
        assert_eq!(bin_i32(BinOp::Shl, 1, 33), Ok(2));
        assert_eq!(bin_i32(BinOp::ShrU, i32::MIN, 31), Ok(1));
        assert_eq!(bin_i32(BinOp::ShrS, i32::MIN, 31), Ok(-1));
        assert_eq!(bin_i64(BinOp::Shl, 1, 64), Ok(1));
        assert_eq!(bin_i64(BinOp::Rotl, 1, 63), Ok(i64::MIN));
        assert_eq!(bin_i32(BinOp::Rotr, 1, 1), Ok(i32::MIN));
    }

    #[test]
    fn narrow_sign_extends() {
        assert_eq!(un_i32(IntUnOp::Extend8S, 200), -56);
        assert_eq!(un_i32(IntUnOp::Extend16S, 0x8000), -32768);
        assert_eq!(un_i32(IntUnOp::Extend32S, -5), -5);
        assert_eq!(un_i64(IntUnOp::Extend32S, 0x8000_0000), -0x8000_0000);
        assert_eq!(un_i64(IntUnOp::Extend8S, 0x17f), 0x7f);
    }

    #[test]
    fn reinterpret_moves_bits_exactly() {
        assert_eq!(
            cast(CastOp::ReinterpI32F32, SpecVal::I32(0x7f80_0001u32 as i32)),
            SpecVal::F32(0x7f80_0001) // sNaN bits pass through a pure bit move
        );
        assert_eq!(
            cast(CastOp::ReinterpF64I64, SpecVal::F64(0x8000_0000_0000_0000)),
            SpecVal::I64(i64::MIN)
        );
    }

    #[test]
    fn demote_rounds_ties_to_even() {
        // 1 + 2^-24 is exactly halfway between f32(1.0) and the next f32 up → 1.0.
        assert_eq!(
            cast(CastOp::Demote, SpecVal::F64(0x3ff0_0000_1000_0000)),
            SpecVal::F32(0x3f80_0000)
        );
        // 1 + 3·2^-24 is halfway between 1+2^-23 and 1+2^-22 → the even one (1+2^-22).
        assert_eq!(
            cast(CastOp::Demote, SpecVal::F64(0x3ff0_0000_3000_0000)),
            SpecVal::F32(0x3f80_0002)
        );
        // 2^128 overflows f32 → +inf.
        assert_eq!(
            cast(CastOp::Demote, SpecVal::F64(0x47f0_0000_0000_0000)),
            SpecVal::F32(0x7f80_0000)
        );
    }

    /// The float definitional cases the prose is terse about, pinned as spec-internal
    /// expectations: min/max NaN propagation and zero ordering, nearest ties-to-even,
    /// and the exact trapping-conversion boundary lattice.
    #[test]
    fn float_definitional_cases() {
        // min/max propagate NaN (wasm fmin/fmax, NOT IEEE minNum) and order -0 < +0.
        assert!(fbin_f32(FBinOp::Min, f32::NAN, 5.0).is_nan());
        assert!(fbin_f64(FBinOp::Max, 5.0, f64::NAN).is_nan());
        assert_eq!(fbin_f32(FBinOp::Min, 0.0, -0.0).to_bits(), 0x8000_0000);
        assert_eq!(fbin_f32(FBinOp::Max, -0.0, 0.0).to_bits(), 0x0000_0000);
        // nearest = round ties to even.
        assert_eq!(fun_f32(FUnOp::Nearest, 2.5), 2.0);
        assert_eq!(fun_f32(FUnOp::Nearest, 3.5), 4.0);
        assert_eq!(fun_f64(FUnOp::Nearest, -0.5).to_bits(), (-0.0f64).to_bits());
        // ne is the negation of eq, so it is TRUE on NaN.
        assert_eq!(fcmp(FCmpOp::Ne, f64::NAN, f64::NAN), 1);
        assert_eq!(fcmp(FCmpOp::Eq, f64::NAN, f64::NAN), 0);
        // Trapping conversion bounds: exact float boundaries, strict on the open side.
        let f64v = SpecVal::F64;
        assert_eq!(
            trunc_trap(FToI::F64I32S, f64v(0x41df_ffff_ffe0_0000)), // 2147483647.5
            Ok(SpecVal::I32(i32::MAX))
        );
        assert_eq!(
            trunc_trap(FToI::F64I32S, f64v(0x41e0_0000_0000_0000)), // 2^31
            Err(SpecTrap::BadConversion)
        );
        assert_eq!(
            trunc_trap(FToI::F64I64S, f64v(0xc3e0_0000_0000_0000)), // -2^63 exact
            Ok(SpecVal::I64(i64::MIN))
        );
        assert_eq!(
            trunc_trap(FToI::F64I32U, f64v((-0.5f64).to_bits())),
            Ok(SpecVal::I32(0)) // trunc(-0.5) = -0 → 0 fits unsigned
        );
        assert_eq!(
            trunc_trap(FToI::F64I32U, f64v((-1.0f64).to_bits())),
            Err(SpecTrap::BadConversion)
        );
        // Saturating: NaN → 0, out-of-range clamps.
        assert_eq!(
            trunc_sat(FToI::F32I32S, SpecVal::F32(0x7fc0_0000)),
            SpecVal::I32(0)
        );
        assert_eq!(
            trunc_sat(FToI::F32I32S, SpecVal::F32(0xff80_0000)), // -inf
            SpecVal::I32(i32::MIN)
        );
        assert_eq!(
            trunc_sat(FToI::F32I64U, SpecVal::F32(0x5f80_0000)), // 2^64
            SpecVal::I64(-1)                                     // u64::MAX
        );
    }

    /// The window-model definitional lattice (§4 / TRAP_CONFINEMENT.md): the whole
    /// span within `[0, mapped)`, computed without wraparound — a wrapping effective
    /// address faults, never aliases; zero-length bulk ops are no-ops at wild
    /// pointers; a faulting access mutates nothing.
    #[test]
    fn window_model_lattice() {
        // Scalar admission boundaries.
        assert_eq!(
            admit(MEM_SIZE - 4, 0, 4, MEM_SIZE),
            Ok((MEM_SIZE - 4) as usize)
        );
        assert_eq!(
            admit(MEM_SIZE - 3, 0, 4, MEM_SIZE),
            Err(SpecTrap::MemoryFault)
        );
        assert_eq!(admit(MEM_SIZE, 0, 1, MEM_SIZE), Err(SpecTrap::MemoryFault));
        assert_eq!(
            admit(0, MEM_SIZE - 8, 8, MEM_SIZE),
            Ok((MEM_SIZE - 8) as usize)
        );
        // Wrap-around faults — the mathematical sum is out of range even though the
        // wrapped u64 would land in-window.
        assert_eq!(admit(u64::MAX, 2, 1, MEM_SIZE), Err(SpecTrap::MemoryFault));
        assert_eq!(admit(2, u64::MAX, 1, MEM_SIZE), Err(SpecTrap::MemoryFault));
        assert_eq!(admit(u64::MAX, 0, 8, MEM_SIZE), Err(SpecTrap::MemoryFault));
        // Bulk spans: zero length is inert anywhere; oversized/overflowing lengths fault.
        assert_eq!(admit_span(u64::MAX, 0, MEM_SIZE), Ok(()));
        assert_eq!(admit_span(0, MEM_SIZE, MEM_SIZE), Ok(()));
        assert_eq!(
            admit_span(0, MEM_SIZE + 1, MEM_SIZE),
            Err(SpecTrap::MemoryFault)
        );
        assert_eq!(
            admit_span(1, u64::MAX, MEM_SIZE),
            Err(SpecTrap::MemoryFault)
        );
        // Loads sign/zero-extend per op; stores write the low bytes little-endian.
        let mut win = vec![0u8; MEM_SIZE as usize];
        win[8] = 0x80;
        assert_eq!(
            load_eval(LoadOp::I32_8S, 8, 0, &win),
            Ok(SpecVal::I32(-128))
        );
        assert_eq!(load_eval(LoadOp::I32_8U, 8, 0, &win), Ok(SpecVal::I32(128)));
        store_eval(StoreOp::I64_32, 0, 0, SpecVal::I64(-1), &mut win).unwrap();
        assert_eq!(&win[0..5], &[0xff, 0xff, 0xff, 0xff, 0x00]);
        // A faulting store mutates nothing.
        let before = win.clone();
        assert_eq!(
            store_eval(StoreOp::I64, MEM_SIZE - 4, 0, SpecVal::I64(-1), &mut win),
            Err(SpecTrap::MemoryFault)
        );
        assert_eq!(win, before);
    }

    /// Table invariants: unique ids, one result each, every row's class agrees with
    /// [`coverage`] on the instruction it builds, and the landed slices have the
    /// expected shape.
    #[test]
    fn table_invariants() {
        assert_eq!(scalar_rows().len(), 80, "slice-1 row count");
        assert_eq!(float_rows().len(), 70, "slice-2 row count");
        let rows = all_rows();
        let mut ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), rows.len(), "duplicate row id");
        for row in &rows {
            let idx: Vec<ValIdx> = (0..row.operands.len() as ValIdx).collect();
            let sample: Vec<SpecVal> = row.inputs().into_iter().map(|t| pool(t)[0]).collect();
            let inst = (row.build)(&idx, &sample);
            assert_eq!(coverage(&inst), row.class, "class mismatch for {}", row.id);
            // Every vector the suite will feed this row type-checks against its inputs.
            for v in vectors_for(row).iter().take(8) {
                for (val, want) in v.iter().zip(row.inputs()) {
                    assert_eq!(val.ty(), want, "pool type mismatch for {}", row.id);
                }
            }
        }
    }

    /// Unary and binary rows must get their FULL boundary cross product — striding is
    /// reserved for wider arities (see `VECTOR_CAP`).
    #[test]
    fn no_striding_below_ternary() {
        for row in all_rows() {
            let n: usize = row.inputs().iter().map(|t| pool(*t).len()).product();
            if row.inputs().len() <= 2 {
                assert_eq!(vectors_for(&row).len(), n, "strided binary row {}", row.id);
            }
        }
    }
}
