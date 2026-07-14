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
//! Slice 1 (SPEC.md implementation plan): the scalar integer core — consts,
//! `IntBin`/`IntCmp`/`IntUn`/`Eqz`/`Convert`/`Select`, `Cast`, `PtrAdd`/`PtrCast`.

#![forbid(unsafe_code)]

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
        build: Box::new(|_, imm| Inst::ConstI32(as_i32(imm[0]))),
        eval: Box::new(|x| Ok(x[0])),
    });
    push(OpRow {
        id: "i64.const".into(),
        operands: vec![],
        result: ValType::I64,
        class: Class::Pure,
        shape: Shape::Immediate,
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
            build: Box::new(move |v, _| Inst::PtrCast { to_int, a: v[0] }),
            eval: Box::new(|x| Ok(x[0])),
        });
    }

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
    0x4f00_0000, // 2^31
    0x5f00_0000, // 2^63
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
    0x43e0_0000_0000_0000, // 2^63
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

    /// Table invariants: unique ids, one result each, every row's class agrees with
    /// [`coverage`] on the instruction it builds, and slice 1 has the expected shape.
    #[test]
    fn table_invariants() {
        let rows = scalar_rows();
        assert_eq!(rows.len(), 80, "slice-1 row count (update on new rows)");
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
        for row in scalar_rows() {
            let n: usize = row.inputs().iter().map(|t| pool(*t).len()).product();
            if row.inputs().len() <= 2 {
                assert_eq!(vectors_for(&row).len(), n, "strided binary row {}", row.id);
            }
        }
    }
}
