//! **SVM IR → WebAssembly emitter** — slice 1 of the browser wasm-JIT tier (`BROWSER.md`
//! § "wasm-JIT tier — design & implementation plan").
//!
//! Compiles a verified [`svm_ir::Module`]'s functions into one WebAssembly module (binary bytes,
//! hand-encoded — no dependencies, like the other escape-TCB-adjacent crates), so hot guest compute
//! can run on a wasm engine's optimizing tiers instead of the bytecode dispatch loop. **Fail-closed
//! like `svm-jit`**: anything outside the supported subset returns [`Error::Unsupported`] and the
//! caller keeps the function on the bytecode-interpreter tier.
//!
//! ## Emitted shape
//!
//! One exported wasm function `f{i}` per SVM function `i`. Each takes two prepended environment
//! params ahead of the SVM signature:
//!
//! - `win: i32` — the guest window's base address in linear memory (import `env.memory`);
//! - `env: i32` — the engine-side environment cell: an `i64` **fuel counter** at offset 0, debited
//!   once per dispatcher iteration (coarser than the interpreter's per-op fuel — a bound, not an
//!   observable; DESIGN.md §5) and trapped on exhaustion.
//!
//! ## Confinement (the load-bearing part)
//!
//! Every guest access replicates the trap-confinement `svm_mask::Window::checked` **exactly**
//! (§4, D38): with `mask = (1 << DEFAULT_RESERVED_LOG2) - 1` and `mapped = 1 << size_log2`,
//!
//! ```text
//! eff = addr + offset;   if eff > mapped - width { trap(MemoryFault) }   // unmasked check
//! access linear memory at win + (eff & mask)   // clamp: no-op past the check
//! ```
//!
//! both constants baked at compile time. An out-of-window address **faults at the offending
//! access** — it is never wrapped back into the window (the `& mask` after the check mirrors the
//! native JIT's check+clamp lowering and cannot change a passing address). SVM-specific traps (memory fault, fuel) route through the
//! imported `env.trap(code)` (the host records the code; the following `unreachable` aborts);
//! div/rem-by-zero, signed-overflow, and `unreachable` map to wasm's own identical traps.
//!
//! ## Control flow
//!
//! v1 is the **block dispatcher**: SSA values live in wasm locals (one per block-scoped value —
//! block params first, then each instruction's results, mirroring the verifier's numbering); a
//! `loop` re-dispatches on a `$next` local via `br_table` over one wasm `block` per SVM block.
//! Branch arguments are pushed onto the operand stack *then* popped into the target's param locals
//! (reverse order), so a self-branch that permutes its own params can't read an already-overwritten
//! local. A relooper for reducible CFGs is a planned upgrade, not a correctness need.
//!
//! Proven by `tests/differential.rs`: every kernel runs on the bytecode engine (the oracle) and on
//! the emitted wasm under `wasmi`, comparing results **and trap kinds**.

#![forbid(unsafe_code)]

use svm_ir::{
    AtomicRmwOp, BinOp, Block, CmpOp, ConvOp, Func, FuncType, Inst, IntTy, IntUnOp, LoadOp, Module,
    StoreOp, Terminator, ValType, DEFAULT_RESERVED_LOG2,
};

/// Trap code delivered through `env.trap` when the per-dispatch fuel counter goes negative.
pub const TRAP_OUT_OF_FUEL: i32 = 1;
/// Trap code delivered through `env.trap` when an access fails the trap-confinement bounds
/// check (`addr + offset + width > mapped` — the §4 `MemoryFault` at the offending access).
pub const TRAP_MEMORY_FAULT: i32 = 2;

/// Why a module was refused. Fail-closed: the caller runs the module on the interpreter tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// An instruction / terminator / type outside the v1 subset (the payload names it).
    Unsupported(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "unsupported by the wasm tier: {what}"),
        }
    }
}

const MASK: u64 = (1u64 << DEFAULT_RESERVED_LOG2) - 1;

// ---- wasm binary encoding primitives -------------------------------------------------------------

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn sleb64(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        if done {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn sleb32(out: &mut Vec<u8>, v: i32) {
    sleb64(out, v as i64);
}

fn valtype_byte(t: ValType) -> Result<u8, Error> {
    match t {
        ValType::I32 => Ok(0x7f),
        ValType::I64 => Ok(0x7e),
        ValType::F32 => Ok(0x7d),
        ValType::F64 => Ok(0x7c),
        ValType::V128 => Ok(0x7b),
        // ref types are interpreter-tier (no funcref/externref values in the emitted subset).
        _ => Err(Error::Unsupported("ref value type")),
    }
}

// A handful of opcode groups the emitter uses; everything else is written as raw bytes at the
// emission site with a comment.
const OP_UNREACHABLE: u8 = 0x00;
const OP_LOOP: u8 = 0x03;
const OP_IF: u8 = 0x04;
const OP_ELSE: u8 = 0x05;
const OP_END: u8 = 0x0b;
const OP_BR: u8 = 0x0c;
const OP_BR_TABLE: u8 = 0x0e;
const OP_RETURN: u8 = 0x0f;
const OP_CALL: u8 = 0x10;
// Tail-call proposal (shipped in V8 ≥ Chrome 112, wasmi ≥ 0.47, Wasmtime): a true tail call that
// **reuses the caller's frame** (O(1) stack), matching the interpreter's frame-reusing `Op::TailCall`.
const OP_RETURN_CALL: u8 = 0x12;
const OP_RETURN_CALL_INDIRECT: u8 = 0x13;
const OP_BLOCK: u8 = 0x02;
const OP_LOCAL_GET: u8 = 0x20;
const OP_LOCAL_SET: u8 = 0x21;
const OP_LOCAL_TEE: u8 = 0x22;
const OP_SELECT: u8 = 0x1b;
const OP_I32_CONST: u8 = 0x41;
const OP_I64_CONST: u8 = 0x42;
const BLOCKTYPE_VOID: u8 = 0x40;

/// `IntBin` opcodes are contiguous in wasm in exactly [`BinOp`]'s declaration order.
fn intbin_opcode(ty: IntTy, op: BinOp) -> u8 {
    let idx = BinOp::ALL.iter().position(|o| *o == op).unwrap() as u8;
    match ty {
        IntTy::I32 => 0x6a + idx,
        IntTy::I64 => 0x7c + idx,
    }
}

fn intcmp_opcode(ty: IntTy, op: CmpOp) -> u8 {
    // wasm orders lt/gt before le/ge; CmpOp orders le before gt — map explicitly.
    let (i32op, i64op) = match op {
        CmpOp::Eq => (0x46, 0x51),
        CmpOp::Ne => (0x47, 0x52),
        CmpOp::LtS => (0x48, 0x53),
        CmpOp::LtU => (0x49, 0x54),
        CmpOp::LeS => (0x4c, 0x57),
        CmpOp::LeU => (0x4d, 0x58),
        CmpOp::GtS => (0x4a, 0x55),
        CmpOp::GtU => (0x4b, 0x56),
        CmpOp::GeS => (0x4e, 0x59),
        CmpOp::GeU => (0x4f, 0x5a),
    };
    match ty {
        IntTy::I32 => i32op,
        IntTy::I64 => i64op,
    }
}

fn intun_opcode(ty: IntTy, op: IntUnOp) -> Result<u8, Error> {
    Ok(match (ty, op) {
        (IntTy::I32, IntUnOp::Clz) => 0x67,
        (IntTy::I32, IntUnOp::Ctz) => 0x68,
        (IntTy::I32, IntUnOp::Popcnt) => 0x69,
        (IntTy::I32, IntUnOp::Extend8S) => 0xc0,
        (IntTy::I32, IntUnOp::Extend16S) => 0xc1,
        (IntTy::I32, IntUnOp::Extend32S) => {
            return Err(Error::Unsupported("i32.extend32_s"));
        }
        (IntTy::I64, IntUnOp::Clz) => 0x79,
        (IntTy::I64, IntUnOp::Ctz) => 0x7a,
        (IntTy::I64, IntUnOp::Popcnt) => 0x7b,
        (IntTy::I64, IntUnOp::Extend8S) => 0xc2,
        (IntTy::I64, IntUnOp::Extend16S) => 0xc3,
        (IntTy::I64, IntUnOp::Extend32S) => 0xc4,
    })
}

// ---- scalar float opcodes (all map 1:1 to core wasm; `Fma` has no core-wasm scalar op) ----------

fn fbin_opcode(ty: svm_ir::FloatTy, op: svm_ir::FBinOp) -> u8 {
    use svm_ir::FBinOp::*;
    // wasm f32 add..copysign are 0x92..0x98, f64 add..copysign 0xa0..0xa6, in FBinOp's exact order.
    let idx = match op {
        Add => 0,
        Sub => 1,
        Mul => 2,
        Div => 3,
        Min => 4,
        Max => 5,
        Copysign => 6,
    };
    match ty {
        svm_ir::FloatTy::F32 => 0x92 + idx,
        svm_ir::FloatTy::F64 => 0xa0 + idx,
    }
}

fn fun_opcode(ty: svm_ir::FloatTy, op: svm_ir::FUnOp) -> u8 {
    use svm_ir::FUnOp::*;
    // wasm orders abs neg ceil floor trunc nearest sqrt; FUnOp orders sqrt before ceil — map explicitly.
    let (f32op, f64op) = match op {
        Abs => (0x8b, 0x99),
        Neg => (0x8c, 0x9a),
        Ceil => (0x8d, 0x9b),
        Floor => (0x8e, 0x9c),
        Trunc => (0x8f, 0x9d),
        Nearest => (0x90, 0x9e),
        Sqrt => (0x91, 0x9f),
    };
    match ty {
        svm_ir::FloatTy::F32 => f32op,
        svm_ir::FloatTy::F64 => f64op,
    }
}

fn fcmp_opcode(ty: svm_ir::FloatTy, op: svm_ir::FCmpOp) -> u8 {
    use svm_ir::FCmpOp::*;
    // wasm orders eq ne lt gt le ge; FCmpOp orders le before gt — map explicitly.
    let (f32op, f64op) = match op {
        Eq => (0x5b, 0x61),
        Ne => (0x5c, 0x62),
        Lt => (0x5d, 0x63),
        Le => (0x5f, 0x65),
        Gt => (0x5e, 0x64),
        Ge => (0x60, 0x66),
    };
    match ty {
        svm_ir::FloatTy::F32 => f32op,
        svm_ir::FloatTy::F64 => f64op,
    }
}

/// `i32/i64.trunc_sat_f32/f64_{s,u}` — the `0xFC` prefix + subopcode (saturating float→int).
fn ftoisat_subop(op: svm_ir::FToI) -> u8 {
    let (fty, ity, signed) = op.parts();
    // subopcode = (int:i32=0/i64=4) + (float:f32=0/f64=2) + (signed?0:1).
    let base = match ity {
        IntTy::I32 => 0,
        IntTy::I64 => 4,
    } + match fty {
        svm_ir::FloatTy::F32 => 0,
        svm_ir::FloatTy::F64 => 2,
    };
    base + if signed { 0 } else { 1 }
}

/// `i32/i64.trunc_f32/f64_{s,u}` — the trapping float→int opcodes (NaN / out-of-range trap).
fn ftoitrap_opcode(op: svm_ir::FToI) -> u8 {
    let (fty, ity, signed) = op.parts();
    match (ity, fty, signed) {
        (IntTy::I32, svm_ir::FloatTy::F32, true) => 0xa8,
        (IntTy::I32, svm_ir::FloatTy::F32, false) => 0xa9,
        (IntTy::I32, svm_ir::FloatTy::F64, true) => 0xaa,
        (IntTy::I32, svm_ir::FloatTy::F64, false) => 0xab,
        (IntTy::I64, svm_ir::FloatTy::F32, true) => 0xae,
        (IntTy::I64, svm_ir::FloatTy::F32, false) => 0xaf,
        (IntTy::I64, svm_ir::FloatTy::F64, true) => 0xb0,
        (IntTy::I64, svm_ir::FloatTy::F64, false) => 0xb1,
    }
}

/// `f32/f64.convert_i32/i64_{s,u}` — int→float.
fn itof_opcode(op: svm_ir::IToF) -> u8 {
    let (ity, fty, signed) = op.parts();
    match (fty, ity, signed) {
        (svm_ir::FloatTy::F32, IntTy::I32, true) => 0xb2,
        (svm_ir::FloatTy::F32, IntTy::I32, false) => 0xb3,
        (svm_ir::FloatTy::F32, IntTy::I64, true) => 0xb4,
        (svm_ir::FloatTy::F32, IntTy::I64, false) => 0xb5,
        (svm_ir::FloatTy::F64, IntTy::I32, true) => 0xb7,
        (svm_ir::FloatTy::F64, IntTy::I32, false) => 0xb8,
        (svm_ir::FloatTy::F64, IntTy::I64, true) => 0xb9,
        (svm_ir::FloatTy::F64, IntTy::I64, false) => 0xba,
    }
}

/// `demote`/`promote`/`reinterpret` cast opcode.
fn cast_opcode(op: svm_ir::CastOp) -> u8 {
    use svm_ir::CastOp::*;
    match op {
        Demote => 0xb6,
        Promote => 0xbb,
        ReinterpI32F32 => 0xbe,
        ReinterpF32I32 => 0xbc,
        ReinterpI64F64 => 0xbf,
        ReinterpF64I64 => 0xbd,
    }
}

/// `(opcode, access width, result type)` for a load.
fn load_op(op: LoadOp) -> Result<(u8, u64, ValType), Error> {
    Ok(match op {
        LoadOp::I32 => (0x28, 4, ValType::I32),
        LoadOp::I64 => (0x29, 8, ValType::I64),
        LoadOp::F32 => (0x2a, 4, ValType::F32),
        LoadOp::F64 => (0x2b, 8, ValType::F64),
        LoadOp::I32_8S => (0x2c, 1, ValType::I32),
        LoadOp::I32_8U => (0x2d, 1, ValType::I32),
        LoadOp::I32_16S => (0x2e, 2, ValType::I32),
        LoadOp::I32_16U => (0x2f, 2, ValType::I32),
        LoadOp::I64_8S => (0x30, 1, ValType::I64),
        LoadOp::I64_8U => (0x31, 1, ValType::I64),
        LoadOp::I64_16S => (0x32, 2, ValType::I64),
        LoadOp::I64_16U => (0x33, 2, ValType::I64),
        LoadOp::I64_32S => (0x34, 4, ValType::I64),
        LoadOp::I64_32U => (0x35, 4, ValType::I64),
    })
}

/// `(plain-load opcode, plain-store opcode, access width)` for a §12 atomic of integer type `ty`.
/// Atomics are 4- or 8-byte only (`atomic_width`), so the plain `i32`/`i64` load/store carry them.
fn atomic_ops(ty: IntTy) -> (u8, u8, u64) {
    match ty {
        IntTy::I32 => (0x28, 0x36, 4), // i32.load, i32.store
        IntTy::I64 => (0x29, 0x37, 8), // i64.load, i64.store
    }
}

/// The arithmetic/bitwise `BinOp` an atomic RMW applies (all but `Xchg`, handled separately).
fn rmw_binop(op: AtomicRmwOp) -> BinOp {
    match op {
        AtomicRmwOp::Add => BinOp::Add,
        AtomicRmwOp::Sub => BinOp::Sub,
        AtomicRmwOp::And => BinOp::And,
        AtomicRmwOp::Or => BinOp::Or,
        AtomicRmwOp::Xor => BinOp::Xor,
        AtomicRmwOp::Xchg => unreachable!("xchg is lowered without a binop"),
    }
}

/// `(opcode, access width)` for a store.
fn store_op(op: StoreOp) -> Result<(u8, u64), Error> {
    Ok(match op {
        StoreOp::I32 => (0x36, 4),
        StoreOp::I64 => (0x37, 8),
        StoreOp::F32 => (0x38, 4),
        StoreOp::F64 => (0x39, 8),
        StoreOp::I32_8 => (0x3a, 1),
        StoreOp::I32_16 => (0x3b, 2),
        StoreOp::I64_8 => (0x3c, 1),
        StoreOp::I64_16 => (0x3d, 2),
        StoreOp::I64_32 => (0x3e, 4),
    })
}

// ---- §17 SIMD (v128) opcodes -------------------------------------------------------------------
//
// Every core-wasm SIMD op is the `0xFD` prefix + a uleb128 subopcode (many are ≥128, so 2 bytes).
// These helpers return the subopcode; [`emit_simd`] writes the prefix + uleb. The numbers are the
// finalized (fixed-128, non-relaxed) wasm SIMD assignments; the exhaustive `tests/simd.rs`
// differential re-derives every one against the bytecode oracle, so a wrong number can't slip
// through (wasmi rejects an invalid encoding, or the lane result diverges).
//
// Deferred to a later increment (fail-closed here → the module stays on the interpreter): the
// widening / reduction family (`extend`/`narrow`/`extmul`/`extadd_pairwise`/`dot`/`q15mulr`) and
// **relaxed** SIMD (`VFma`, `VDotI8` — no core-wasm opcode, like scalar `Fma`).

use svm_ir::{
    VFCmpOp, VFloatBinOp, VFloatUnOp, VICmpOp, VIntBinOp, VIntUnOp, VNarrowOp, VPMinMaxOp,
    VSatBinOp, VShape, VShiftOp, VWidenOp,
};

const OP_SIMD_PREFIX: u8 = 0xfd;

/// Write a SIMD instruction: the `0xFD` prefix + the uleb subopcode.
fn emit_simd(code: &mut Vec<u8>, sub: u32) {
    code.push(OP_SIMD_PREFIX);
    uleb(code, sub as u64);
}

/// `<shape>.splat` subopcode.
fn vsplat_sub(shape: VShape) -> u32 {
    match shape {
        VShape::I8x16 => 15,
        VShape::I16x8 => 16,
        VShape::I32x4 => 17,
        VShape::I64x2 => 18,
        VShape::F32x4 => 19,
        VShape::F64x2 => 20,
    }
}

/// `<shape>.extract_lane[_s/_u]` subopcode (narrow int shapes carry the sign choice).
fn vextract_sub(shape: VShape, signed: bool) -> u32 {
    match shape {
        VShape::I8x16 => {
            if signed {
                21
            } else {
                22
            }
        }
        VShape::I16x8 => {
            if signed {
                24
            } else {
                25
            }
        }
        VShape::I32x4 => 27,
        VShape::I64x2 => 29,
        VShape::F32x4 => 31,
        VShape::F64x2 => 33,
    }
}

/// `<shape>.replace_lane` subopcode.
fn vreplace_sub(shape: VShape) -> u32 {
    match shape {
        VShape::I8x16 => 23,
        VShape::I16x8 => 26,
        VShape::I32x4 => 28,
        VShape::I64x2 => 30,
        VShape::F32x4 => 32,
        VShape::F64x2 => 34,
    }
}

/// Lane-wise integer binary op subopcode (`None` for the holes wasm omits: `i8x16.mul`, `i64x2`
/// min/max).
fn vintbin_sub(shape: VShape, op: VIntBinOp) -> Option<u32> {
    use VIntBinOp::*;
    Some(match (shape, op) {
        (VShape::I8x16, Add) => 110,
        (VShape::I8x16, Sub) => 113,
        (VShape::I8x16, MinS) => 118,
        (VShape::I8x16, MinU) => 119,
        (VShape::I8x16, MaxS) => 120,
        (VShape::I8x16, MaxU) => 121,
        (VShape::I16x8, Add) => 142,
        (VShape::I16x8, Sub) => 145,
        (VShape::I16x8, Mul) => 149,
        (VShape::I16x8, MinS) => 150,
        (VShape::I16x8, MinU) => 151,
        (VShape::I16x8, MaxS) => 152,
        (VShape::I16x8, MaxU) => 153,
        (VShape::I32x4, Add) => 174,
        (VShape::I32x4, Sub) => 177,
        (VShape::I32x4, Mul) => 181,
        (VShape::I32x4, MinS) => 182,
        (VShape::I32x4, MinU) => 183,
        (VShape::I32x4, MaxS) => 184,
        (VShape::I32x4, MaxU) => 185,
        (VShape::I64x2, Add) => 206,
        (VShape::I64x2, Sub) => 209,
        (VShape::I64x2, Mul) => 213,
        // i8x16.mul and i64x2 min/max have no wasm opcode.
        _ => return None,
    })
}

/// Lane-wise integer comparison subopcode (`i64x2` has only signed `eq`/`ne`/`lt`/`gt`/`le`/`ge`).
fn vintcmp_sub(shape: VShape, op: VICmpOp) -> Option<u32> {
    use VICmpOp::*;
    let base = match shape {
        VShape::I8x16 => 35,
        VShape::I16x8 => 45,
        VShape::I32x4 => 55,
        VShape::I64x2 => {
            return Some(match op {
                Eq => 214,
                Ne => 215,
                LtS => 216,
                GtS => 217,
                LeS => 218,
                GeS => 219,
                // i64x2 has no unsigned lane compares in wasm.
                LtU | GtU | LeU | GeU => return None,
            });
        }
        VShape::F32x4 | VShape::F64x2 => return None,
    };
    Some(base + op.index() as u32)
}

/// Lane-wise float comparison subopcode.
fn vfloatcmp_sub(shape: VShape, op: VFCmpOp) -> Option<u32> {
    let base = match shape {
        VShape::F32x4 => 65,
        VShape::F64x2 => 71,
        _ => return None,
    };
    Some(base + op.index() as u32)
}

/// Lane-wise integer shift subopcode (integer shapes only).
fn vshift_sub(shape: VShape, op: VShiftOp) -> Option<u32> {
    let base = match shape {
        VShape::I8x16 => 107,
        VShape::I16x8 => 139,
        VShape::I32x4 => 171,
        VShape::I64x2 => 203,
        _ => return None,
    };
    Some(base + op.index() as u32)
}

/// Lane-wise unary integer op subopcode (`abs`/`neg`, every integer shape).
fn vintun_sub(shape: VShape, op: VIntUnOp) -> Option<u32> {
    let base = match shape {
        VShape::I8x16 => 96,
        VShape::I16x8 => 128,
        VShape::I32x4 => 160,
        VShape::I64x2 => 192,
        _ => return None,
    };
    Some(base + op.index() as u32) // Abs=+0, Neg=+1
}

/// Saturating add/sub subopcode (`i8x16`/`i16x8` only).
fn vsatbin_sub(shape: VShape, op: VSatBinOp) -> Option<u32> {
    use VSatBinOp::*;
    Some(match (shape, op) {
        (VShape::I8x16, AddS) => 111,
        (VShape::I8x16, AddU) => 112,
        (VShape::I8x16, SubS) => 114,
        (VShape::I8x16, SubU) => 115,
        (VShape::I16x8, AddS) => 143,
        (VShape::I16x8, AddU) => 144,
        (VShape::I16x8, SubS) => 146,
        (VShape::I16x8, SubU) => 147,
        _ => return None,
    })
}

/// `<shape>.avgr_u` subopcode (`i8x16`/`i16x8` only).
fn vavgr_sub(shape: VShape) -> Option<u32> {
    match shape {
        VShape::I8x16 => Some(123),
        VShape::I16x8 => Some(155),
        _ => None,
    }
}

/// `<shape>.all_true` subopcode (integer shapes).
fn valltrue_sub(shape: VShape) -> Option<u32> {
    match shape {
        VShape::I8x16 => Some(99),
        VShape::I16x8 => Some(131),
        VShape::I32x4 => Some(163),
        VShape::I64x2 => Some(195),
        _ => None,
    }
}

/// `<shape>.bitmask` subopcode (integer shapes).
fn vbitmask_sub(shape: VShape) -> Option<u32> {
    match shape {
        VShape::I8x16 => Some(100),
        VShape::I16x8 => Some(132),
        VShape::I32x4 => Some(164),
        VShape::I64x2 => Some(196),
        _ => None,
    }
}

/// Lane-wise binary float op subopcode.
fn vfloatbin_sub(shape: VShape, op: VFloatBinOp) -> Option<u32> {
    let base = match shape {
        VShape::F32x4 => 228,
        VShape::F64x2 => 240,
        _ => return None,
    };
    Some(base + op.index() as u32) // Add..Max contiguous
}

/// Lane-wise unary float op subopcode (abs/neg/sqrt regular; ceil/floor/trunc/nearest scattered).
fn vfloatun_sub(shape: VShape, op: VFloatUnOp) -> Option<u32> {
    use VFloatUnOp::*;
    Some(match (shape, op) {
        (VShape::F32x4, Abs) => 224,
        (VShape::F32x4, Neg) => 225,
        (VShape::F32x4, Sqrt) => 227,
        (VShape::F32x4, Ceil) => 103,
        (VShape::F32x4, Floor) => 104,
        (VShape::F32x4, Trunc) => 105,
        (VShape::F32x4, Nearest) => 106,
        (VShape::F64x2, Abs) => 236,
        (VShape::F64x2, Neg) => 237,
        (VShape::F64x2, Sqrt) => 239,
        (VShape::F64x2, Ceil) => 116,
        (VShape::F64x2, Floor) => 117,
        (VShape::F64x2, Trunc) => 122,
        (VShape::F64x2, Nearest) => 148,
        _ => return None,
    })
}

/// Lane-wise pseudo-min/max subopcode (float shapes).
fn vpminmax_sub(shape: VShape, op: VPMinMaxOp) -> Option<u32> {
    use VPMinMaxOp::*;
    Some(match (shape, op) {
        (VShape::F32x4, Pmin) => 234,
        (VShape::F32x4, Pmax) => 235,
        (VShape::F64x2, Pmin) => 246,
        (VShape::F64x2, Pmax) => 247,
        _ => return None,
    })
}

/// Whole-vector bitwise binary op subopcode.
fn vbitbin_sub(op: svm_ir::VBitBinOp) -> u32 {
    use svm_ir::VBitBinOp::*;
    match op {
        And => 78,
        Or => 80,
        Xor => 81,
        AndNot => 79,
    }
}

/// Int↔float / float↔float lane conversion subopcode (all in-subset).
fn vconvert_sub(op: svm_ir::VCvtOp) -> u32 {
    use svm_ir::VCvtOp::*;
    match op {
        F32x4ConvertI32x4S => 250,
        F32x4ConvertI32x4U => 251,
        I32x4TruncSatF32x4S => 248,
        I32x4TruncSatF32x4U => 249,
        F32x4DemoteF64x2Zero => 94,
        F64x2PromoteLowF32x4 => 95,
        F64x2ConvertLowI32x4S => 254,
        F64x2ConvertLowI32x4U => 255,
        I32x4TruncSatF64x2SZero => 252,
        I32x4TruncSatF64x2UZero => 253,
    }
}

// ---- deferred SIMD family (widening / reduction) — added in the simd2 slice ---------------------
//
// wasm lays these out as `{low_s, high_s, low_u, high_u}` contiguously per result shape (a different
// order than [`VWidenOp`]'s `{LowS, LowU, HighS, HighU}`), so map the op to that lane-order offset.
fn widen_lane_offset(op: VWidenOp) -> u32 {
    match op {
        VWidenOp::LowS => 0,
        VWidenOp::HighS => 1,
        VWidenOp::LowU => 2,
        VWidenOp::HighU => 3,
    }
}

/// Lane **widen** (`extend_low/high_<src>_s/u`) subopcode; `shape` is the wider **result** shape.
fn vwiden_sub(shape: VShape, op: VWidenOp) -> Option<u32> {
    let base = match shape {
        VShape::I16x8 => 135, // from i8x16
        VShape::I32x4 => 167, // from i16x8
        VShape::I64x2 => 199, // from i32x4
        _ => return None,
    };
    Some(base + widen_lane_offset(op))
}

/// Lane **narrow** (`narrow_<src>_s/u`) subopcode; `shape` is the narrow **result** shape.
fn vnarrow_sub(shape: VShape, op: VNarrowOp) -> Option<u32> {
    let base = match shape {
        VShape::I8x16 => 101, // from i16x8
        VShape::I16x8 => 133, // from i32x4
        _ => return None,
    };
    Some(base + op.index() as u32) // S=+0, U=+1
}

/// Extended (widening) multiply (`extmul_low/high_<src>_s/u`) subopcode; `shape` is the wide result.
fn vextmul_sub(shape: VShape, op: VWidenOp) -> Option<u32> {
    let base = match shape {
        VShape::I16x8 => 156, // from i8x16
        VShape::I32x4 => 188, // from i16x8
        VShape::I64x2 => 220, // from i32x4
        _ => return None,
    };
    Some(base + widen_lane_offset(op))
}

/// Extended pairwise add (`extadd_pairwise_<src>_s/u`) subopcode; `shape` is the wide result.
fn vextadd_sub(shape: VShape, signed: bool) -> Option<u32> {
    let base = match shape {
        VShape::I16x8 => 124, // from i8x16
        VShape::I32x4 => 126, // from i16x8
        _ => return None,
    };
    Some(base + if signed { 0 } else { 1 })
}

// ---- per-function value typing (mirrors the verifier's block-scoped numbering) -------------------

/// The types of one block's value list: params first, then each instruction's results in order.
/// Only the v1 subset is typed; anything else is `Unsupported` (fail-closed — the module was
/// verified, so no `unwrap` here can be reached by a malformed operand index).
fn block_value_types(m: &Module, b: &Block) -> Result<Vec<ValType>, Error> {
    let mut tys: Vec<ValType> = b.params.clone();
    for inst in &b.insts {
        match inst {
            Inst::ConstI32(_) => tys.push(ValType::I32),
            Inst::ConstI64(_) => tys.push(ValType::I64),
            Inst::IntBin { ty, .. } => tys.push(ty.val()),
            Inst::IntCmp { .. } | Inst::Eqz { .. } => tys.push(ValType::I32),
            Inst::IntUn { ty, .. } => tys.push(ty.val()),
            Inst::Convert { op, .. } => tys.push(op.sig().2),
            Inst::Select { a, .. } => {
                let t = *tys
                    .get(*a as usize)
                    .ok_or(Error::Unsupported("select operand"))?;
                tys.push(t);
            }
            Inst::ConstF32(_) => tys.push(ValType::F32),
            Inst::ConstF64(_) => tys.push(ValType::F64),
            Inst::FBin { ty, .. } | Inst::FUn { ty, .. } => tys.push(ty.val()),
            Inst::FCmp { .. } => tys.push(ValType::I32),
            Inst::FToISat { op, .. } | Inst::FToITrap { op, .. } => tys.push(op.parts().1.val()),
            Inst::IToFConv { op, .. } => tys.push(op.parts().1.val()),
            Inst::Cast { op, .. } => tys.push(op.sig().2),
            // `Fma` has no core-wasm scalar opcode (relaxed-SIMD only), so it stays interpreter-tier.
            Inst::Fma { .. } => return Err(Error::Unsupported("scalar fma (no core-wasm op)")),
            Inst::Load { op, .. } => tys.push(load_op(*op)?.2),
            Inst::Store { .. } => {}
            // §12 atomics lower to a plain load/(rmw)/store sequence plus the interpreter's
            // natural-align trap — observably identical to a hardware atomic **when single-threaded**,
            // and (unlike the core-wasm atomic opcodes, which `wasmi` can't run) differential-testable.
            // The single-thread precondition is enforced module-wide by `func_in_subset`'s
            // `atomics_ok` gate (no concurrency op anywhere ⇒ no contention); here we only type the
            // results. Load/rmw/cmpxchg yield `ty`; store yields nothing.
            Inst::AtomicLoad { ty, .. }
            | Inst::AtomicRmw { ty, .. }
            | Inst::AtomicCmpxchg { ty, .. } => tys.push(ty.val()),
            Inst::AtomicStore { .. } => {}
            // Bulk memory (D62): `memcpy`/`memmove`/`memset` → wasm `memory.copy`/`memory.fill` with
            // whole-span confinement (see the lowering in `emit_block_body`). No SSA result.
            Inst::MemCopy { .. } | Inst::MemMove { .. } | Inst::MemFill { .. } => {}
            Inst::Call { func, .. } => {
                let callee = m
                    .funcs
                    .get(*func as usize)
                    .ok_or(Error::Unsupported("call target"))?;
                tys.extend(callee.results.iter().copied());
            }
            // A funcref is a plain `i32` (the function index, §3c) — a bare `i32.const`.
            Inst::RefFunc { .. } => tys.push(ValType::I32),
            // Indirect call: results come from the call site's own signature immediate.
            Inst::CallIndirect { ty, .. } => tys.extend(ty.results.iter().copied()),
            // ---- §17 SIMD (v128): the in-subset core lane ops (see the opcode helpers above). Each
            // yields a `v128`, except lane-extract (the shape's scalar), the reductions
            // any/all_true/bitmask (`i32`), and `simd.width_bytes` (`i32`). The verifier already
            // typed these, so the emit-side opcode helpers (which return `None`/`Err` for the
            // shape holes wasm omits) are what actually gate a bogus lowering — here we only need
            // the result type. The deferred widening/reduction/relaxed ops fall through to the
            // `_` arm (Unsupported → the module stays on the interpreter).
            Inst::ConstV128(_)
            | Inst::V128Load { .. }
            | Inst::Splat { .. }
            | Inst::ReplaceLane { .. }
            | Inst::VIntBin { .. }
            | Inst::VIntCmp { .. }
            | Inst::VFloatCmp { .. }
            | Inst::VShift { .. }
            | Inst::VIntUn { .. }
            | Inst::VSatBin { .. }
            | Inst::VConvert { .. }
            | Inst::VPMinMax { .. }
            | Inst::VPopcnt { .. }
            | Inst::VAvgr { .. }
            | Inst::VFloatBin { .. }
            | Inst::VFloatUn { .. }
            | Inst::VBitBin { .. }
            | Inst::VNot { .. }
            | Inst::Bitselect { .. }
            | Inst::Shuffle { .. }
            | Inst::Swizzle { .. }
            // simd2: the widening / reduction family (all yield a `v128`). The two **relaxed** ops
            // (`VFma`/`VDotI8`) have no core-wasm opcode, so they fall through to the `_` arm and stay
            // interpreter-tier.
            | Inst::VWiden { .. }
            | Inst::VNarrow { .. }
            | Inst::VExtMul { .. }
            | Inst::VExtAddPairwise { .. }
            | Inst::VDot { .. }
            | Inst::VQ15MulrSat { .. } => tys.push(ValType::V128),
            Inst::V128Store { .. } => {}
            Inst::ExtractLane { shape, .. } => tys.push(shape.lane_val()),
            Inst::VAnyTrue { .. } | Inst::VAllTrue { .. } | Inst::VBitmask { .. } => {
                tys.push(ValType::I32)
            }
            Inst::SimdWidthBytes => tys.push(ValType::I32),
            _ => return Err(Error::Unsupported("instruction outside the v1 subset")),
        }
    }
    Ok(tys)
}

// ---- tiering analysis (slice 3): which functions the JIT tier emits vs. leaves to the interp ------

/// Per-function tiering classification for a module (`BROWSER.md` § "wasm-JIT tier", slice 3).
/// The JIT tier emits the **in-subset** functions and routes a call to an **interp-callable** one
/// through the engine (a cross-tier call); a guest is `mixed_ok` when func 0 and everything it
/// reaches is one or the other (and nothing reachable suspends — a JITted frame can't unwind, so
/// suspension anywhere forces the whole guest to the interpreter).
#[derive(Clone, Debug)]
pub struct Analysis {
    /// `in_subset[i]` — function `i` is entirely within the integer compute subset the emitter
    /// lowers directly (it becomes an emitted `f{i}`).
    pub in_subset: Vec<bool>,
    /// `interp_leaf[i]` — function `i` is **not** in-subset but is safe to run on the bytecode
    /// engine as a cross-tier leaf: an all-integer signature (so the call ABI marshals only i64
    /// slots), **memory-free**, makes no calls (a true leaf, so no transitive window/state to
    /// share), and no concurrency / capability ops. A JITted caller reaches it via `env.call_interp`.
    pub interp_leaf: Vec<bool>,
    /// `reachable[i]` — function `i` is reachable from func 0 through call edges.
    pub reachable: Vec<bool>,
    /// Every reachable function is in-subset or an interp leaf, func 0 is in-subset, and nothing
    /// reachable uses concurrency — i.e. the guest can run on the JIT tier (with cross-tier calls).
    pub mixed_ok: bool,
}

/// A block terminator the emitter lowers. Tail calls (`return_call`/`return_call_indirect`) are
/// included: they lower to the ordinary call sequence (direct / cross-tier / indirect) leaving the
/// callee's results on the stack, followed by `return` — semantically identical, without frame reuse.
/// (`-O2` produces `return_call` for *any* function whose last statement is a call, so accepting them
/// keeps those hot functions — e.g. Doom's `I_FinishUpdate` — emittable rather than interpreter-tier.)
fn term_in_subset(t: &Terminator) -> bool {
    matches!(
        t,
        Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::BrTable { .. }
            | Terminator::Return(_)
            | Terminator::Unreachable
            | Terminator::ReturnCall { .. }
            | Terminator::ReturnCallIndirect { .. }
    )
}

/// Whether every instruction, terminator, and value type of `f` is in the emitter's integer compute
/// subset — reusing [`block_value_types`] (which errors on any out-of-subset instruction) as the
/// single source of truth, plus a type check (all values i32/i64) and the terminator check.
fn func_in_subset(m: &Module, f: &Func, atomics_ok: bool) -> bool {
    // §12 atomics lower to a **single-threaded** load/(rmw)/store sequence (see the
    // `block_value_types` note): correct only when no contention is possible. `atomics_ok` is the
    // module-level guarantee of that (no reachable concurrency op ⇒ no second thread) — when it does
    // not hold, an atomic-using function stays off the JIT tier so the interpreter runs it with true
    // hardware atomicity (matching the tier-up model, which already routes concurrency to the interp).
    if !atomics_ok && func_uses_atomics(f) {
        return false;
    }
    f.blocks.iter().all(|b| {
        block_value_types(m, b).is_ok_and(|tys| tys.iter().all(|t| valtype_byte(*t).is_ok()))
            && term_in_subset(&b.term)
    })
}

/// Whether `f` contains any §12 atomic op ([`Inst::AtomicLoad`]/`Store`/`Rmw`/`Cmpxchg`).
fn func_uses_atomics(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::AtomicLoad { .. }
                    | Inst::AtomicStore { .. }
                    | Inst::AtomicRmw { .. }
                    | Inst::AtomicCmpxchg { .. }
            )
        })
    })
}

/// The module-level single-thread guarantee that makes the atomics' single-threaded lowering sound:
/// **no** function uses a concurrency op (`thread.spawn`/`cont.*`/`memory.wait/notify`), so no second
/// vCPU can ever run and contend an atomic. A guest that spawns threads fails this, keeping its
/// atomic-using functions on the interpreter (true atomicity) — see [`func_in_subset`].
fn module_atomics_ok(m: &Module) -> bool {
    !m.funcs.iter().any(|f| f.uses_concurrency())
}

/// The function indices `f` calls (direct `Call`s + tail-call terminators — the latter keeps the
/// reachability sound even though a tail call itself isn't emitted).
fn func_callees(f: &Func) -> Vec<u32> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Inst::Call { func, .. } = inst {
                out.push(*func);
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            out.push(*func);
        }
    }
    out
}

/// Whether `f` makes an indirect call (`call_indirect`), which can dispatch to **any** function
/// through the identity funcref table — an edge direct-call reachability can't see.
fn func_uses_indirect(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i, Inst::CallIndirect { .. }))
            || matches!(b.term, Terminator::ReturnCallIndirect { .. })
    })
}

/// Whether `f` is safe to run as a cross-tier interpreter leaf (see [`Analysis::interp_leaf`]).
fn interp_leaf(f: &Func) -> bool {
    let int_sig = f
        .params
        .iter()
        .chain(&f.results)
        .all(|t| matches!(t, ValType::I32 | ValType::I64));
    if !int_sig || f.uses_concurrency() {
        return false;
    }
    f.blocks.iter().all(|b| {
        !matches!(
            b.term,
            Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. }
        ) && b.insts.iter().all(|i| {
            !matches!(
                i,
                // memory ops (a leaf's fresh window would diverge from the shared one),
                Inst::Load { .. }
                        | Inst::Store { .. }
                        | Inst::MemCopy { .. }
                        | Inst::MemMove { .. }
                        | Inst::MemFill { .. }
                        | Inst::AtomicLoad { .. }
                        | Inst::AtomicStore { .. }
                        | Inst::AtomicRmw { .. }
                        | Inst::AtomicCmpxchg { .. }
                        | Inst::V128Load { .. }
                        | Inst::V128Store { .. }
                        // calls (a true leaf only — transitive tiers are a later refinement),
                        | Inst::Call { .. }
                        | Inst::CallIndirect { .. }
                        // and host/capability ops (no powerbox in the cross-tier callback).
                        | Inst::CapCall { .. }
                        | Inst::CallImport { .. }
                        | Inst::ImportAttach { .. }
            )
        })
    })
}

/// Classify every function of a **verified** `m` for tiering rooted at func 0 (see [`Analysis`]).
/// Whether `f`'s signature is entirely `i32`/`i64` — the marshalling the cross-tier `env.call_interp`
/// ABI handles (each arg/result is one i64 scratch slot the callback widens/narrows per the declared
/// type). A function with a `v128`/float parameter or result cannot be reached cross-tier.
fn int_sig(f: &Func) -> bool {
    f.params
        .iter()
        .chain(&f.results)
        .all(|t| matches!(t, ValType::I32 | ValType::I64))
}

pub fn analyze(m: &Module) -> Analysis {
    analyze_from(m, 0)
}

/// Like [`analyze`] but reachability and `mixed_ok` are rooted at `entry` — the function the host
/// will call (the JIT entry). The cross-engine bench runs an arbitrary kernel function, not
/// necessarily func 0.
pub fn analyze_from(m: &Module, entry: u32) -> Analysis {
    let n = m.funcs.len();
    let atomics_ok = module_atomics_ok(m);
    let in_subset: Vec<bool> = m
        .funcs
        .iter()
        .map(|f| func_in_subset(m, f, atomics_ok))
        .collect();
    let interp_leaf: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| !in_subset[i] && interp_leaf(f))
        .collect();

    // Reachability from `entry` through call edges.
    let mut reachable = vec![false; n];
    if (entry as usize) < n {
        let mut stack = vec![entry];
        reachable[entry as usize] = true;
        while let Some(fi) = stack.pop() {
            for c in func_callees(&m.funcs[fi as usize]) {
                if (c as usize) < n && !reachable[c as usize] {
                    reachable[c as usize] = true;
                    stack.push(c);
                }
            }
        }
    }

    // `call_indirect` dispatches through the identity funcref table and can reach **any** function —
    // an edge the direct-call walk above can't follow. If a reachable function makes an indirect
    // call, conservatively treat every function as reachable and require them **all** in-subset: the
    // emitted funcref table populates one slot per function, so an index the interpreter would run
    // must resolve to an emitted target rather than a null slot (which would trap). This is the
    // first-increment restriction (all indirect targets in-subset); cross-tier indirect is a later
    // refinement.
    let has_indirect = (0..n).any(|i| reachable[i] && func_uses_indirect(&m.funcs[i]));
    if has_indirect {
        reachable.iter_mut().for_each(|r| *r = true);
    }

    let mixed_ok = (entry as usize) < n
        && in_subset[entry as usize]
        && if has_indirect {
            (0..n).all(|i| in_subset[i])
        } else {
            (0..n).all(|i| !reachable[i] || in_subset[i] || interp_leaf[i])
        };

    Analysis {
        in_subset,
        interp_leaf,
        reachable,
        mixed_ok,
    }
}

// ---- the emitter ---------------------------------------------------------------------------------

/// Compile every function of a **verified** `m` into one wasm module, importing a **non-shared**
/// `env.memory`. See [`compile_module_shared`] for the browser (threads-build) link target.
pub fn compile_module(m: &Module) -> Result<Vec<u8>, Error> {
    compile_module_with(m, false)
}

/// Like [`compile_module`] but the imported `env.memory` is declared **shared** — the browser
/// wasm-JIT tier links the emitted module against the cdylib's shared linear memory (the threads
/// build), and wasm requires the import's shared flag to match the provided memory's. The emitted
/// code is otherwise byte-identical to the non-shared form (only the memory-import limits differ),
/// so the `compile_module` differential (`tests/differential.rs`, under `wasmi`, which has no
/// shared-memory support) fully covers this variant's codegen.
pub fn compile_module_shared(m: &Module) -> Result<Vec<u8>, Error> {
    compile_module_with(m, true)
}

/// Two imported functions precede every emitted function, so a defined function's wasm index is
/// `IMPORTED_FUNCS + its position among the emitted functions`.
const IMPORTED_FUNCS: u32 = 2;
/// `env.call_interp` scratch: the cross-tier call marshals its i64 arg/result slots starting at
/// this byte offset in the `env` cell (past the `i64` fuel counter at 0). The host must allocate the
/// `env` cell at least [`ENV_CELL_BYTES`] large.
const ENV_SCRATCH_OFF: u64 = 16;
/// Max i64 slots the cross-tier scratch holds (a call with more args-or-results than this is refused
/// — 64 is absurdly generous for a function signature).
const XCALL_MAX_SLOTS: usize = 64;
/// Bytes the host must allocate for the `env` cell: the `i64` fuel counter + the cross-tier scratch.
pub const ENV_CELL_BYTES: usize = ENV_SCRATCH_OFF as usize + XCALL_MAX_SLOTS * 8;

/// Compile every function of a **verified** `m` into one wasm module (whole-module, all-integer).
/// Exports `f{i}` per SVM function; imports `env.memory` (shared iff `shared_memory`), `env.trap`,
/// and `env.call_interp`. Returns [`Error::Unsupported`] if *any* function is outside the v1 subset
/// — for a mixed integer/interp guest use [`compile_module_mixed`].
pub fn compile_module_with(m: &Module, shared_memory: bool) -> Result<Vec<u8>, Error> {
    let a = analyze(m);
    if !a.in_subset.iter().all(|&s| s) {
        return Err(Error::Unsupported(
            "a function is outside the integer subset",
        ));
    }
    let n = m.funcs.len();
    let emitted: Vec<usize> = (0..n).collect();
    let wasm_of: Vec<Option<u32>> = (0..n).map(|i| Some(IMPORTED_FUNCS + i as u32)).collect();
    emit_module(m, shared_memory, &emitted, &wasm_of, &a.interp_leaf)
}

/// Compile a **mixed-tier** guest (`BROWSER.md` § "wasm-JIT tier", slice 3): emit the in-subset
/// functions and route a call to an interp leaf through `env.call_interp` (the engine runs it on the
/// bytecode interpreter — see [`Analysis`]). [`Error::Unsupported`] unless [`Analysis::mixed_ok`].
pub fn compile_module_mixed(m: &Module, shared_memory: bool) -> Result<Vec<u8>, Error> {
    compile_module_mixed_entry(m, 0, shared_memory)
}

/// Like [`compile_module_mixed`] but eligibility is rooted at `entry` (the function the host calls),
/// not func 0 — for a module whose entry kernel isn't func 0 (the cross-engine bench). Every emitted
/// function is still exported as `f{svm_idx}`, so the host calls `f{entry}`.
pub fn compile_module_mixed_entry(
    m: &Module,
    entry: u32,
    shared_memory: bool,
) -> Result<Vec<u8>, Error> {
    let a = analyze_from(m, entry);
    if !a.mixed_ok {
        return Err(Error::Unsupported("guest is not mixed-tier runnable"));
    }
    // Emit the reachable in-subset functions in SVM-index order; each gets the next wasm index.
    // Interp leaves (and unreachable / non-subset functions) get no wasm index — a call to a leaf
    // goes to the import. Restricting to `reachable` keeps an unreachable in-subset function (whose
    // own callees `mixed_ok` never checked) from being emitted with an unroutable call.
    let mut wasm_of: Vec<Option<u32>> = vec![None; m.funcs.len()];
    let mut emitted: Vec<usize> = Vec::new();
    for (i, slot) in wasm_of.iter_mut().enumerate() {
        if a.reachable[i] && a.in_subset[i] {
            *slot = Some(IMPORTED_FUNCS + emitted.len() as u32);
            emitted.push(i);
        }
    }
    emit_module(m, shared_memory, &emitted, &wasm_of, &a.interp_leaf)
}

/// **Cap-call outlining** — hoist every inline `cap.call` into a synthetic single-block wrapper
/// function, rewriting the call site to a plain [`Inst::Call`]. Semantics-preserving (the wrapper
/// does the *identical* `cap.call`), but it moves the host-boundary op out of otherwise-emittable
/// functions: the wrapper has an all-integer signature (a capability handle is `i32`, its op args/
/// results are `i64`), so it is a **cross-tier callable** leaf, while the function that used to hold
/// the `cap.call` becomes pure compute + a `Call` and can now emit. This is the compiler doing, on the
/// IR, what a guest author would do by hand (moving `__vm_host_call` into a `noinline` shim) — so an
/// **unmodified** reactor whose hot `tick` interleaves compute with a once-per-frame `present`/`poll`
/// cap call runs its hot path on emitted wasm, bouncing to the interpreter only at the (rare) cap site.
///
/// Existing [`FuncIdx`](svm_ir::FuncIdx)es are unchanged — wrappers are **appended** — so exports,
/// call sites, the function table, and debug locs (all keyed by the original indices) stay valid. The
/// rewrite is **1:1** at each call site: a `Call` to a wrapper appends exactly the wrapper's results,
/// which equal the `cap.call`'s `sig.results`, so block-local value numbering is preserved (no
/// renumbering — the same property [`svm_ir::resolve_imports`] relies on lowering `CallImport`).
///
/// Runs **after** [`svm_ir::resolve_imports`] (it rewrites concrete `cap.call`s, not named imports),
/// and the transformed module must be the one **both** tiers use: the emitter reads it, and the host's
/// `call_interp` runs the wrapper on the interpreter — the wrapper only exists in the outlined module.
///
/// It outlines three host-boundary ops the same way: [`Inst::CapCall`], [`Inst::CallImport`] (an
/// executable manifest import, IMPORTS.md phase 3 — the wrapper carries the `call.import` to the
/// import-capable interpreter tier, so an import-bearing guest emits without resolution or rewrite),
/// and [`Inst::CapSelfResolve`] (the
/// §7 by-name capability lookup the powerbox `_start` synth uses at startup — `cap.self.resolve`). The
/// latter matters for the on-ramp entry: `_start` is otherwise pure compute + stores, so hoisting its
/// handful of `cap.self.resolve`s into cross-tier wrappers makes func 0 itself emittable — the last
/// thing keeping a QuickJS-scale guest (whose hot interpreter loop is all in-subset) off the wasm tier.
pub fn outline_cap_calls(m: &mut Module) {
    let base = m.funcs.len() as u32;
    let mut wrappers: Vec<Func> = Vec::new();
    for f in &mut m.funcs {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                if let Inst::CapCall {
                    type_id,
                    op,
                    sig,
                    handle,
                    args,
                } = inst
                {
                    let g = base + wrappers.len() as u32;
                    // Wrapper signature: (handle: i32, ...sig.params) -> sig.results.
                    let mut params = Vec::with_capacity(1 + sig.params.len());
                    params.push(ValType::I32);
                    params.extend(sig.params.iter().copied());
                    let nparams = params.len() as u32;
                    // Body: `cap.call` on the wrapper's own params (handle = val 0, args = vals 1..),
                    // then return its results (appended right after the params).
                    let wrapper_args: Vec<u32> = (1..nparams).collect();
                    let ret: Vec<u32> = (nparams..nparams + sig.results.len() as u32).collect();
                    let block = Block {
                        params: params.clone(),
                        insts: vec![Inst::CapCall {
                            type_id: *type_id,
                            op: *op,
                            sig: sig.clone(),
                            handle: 0,
                            args: wrapper_args,
                        }],
                        term: Terminator::Return(ret),
                    };
                    wrappers.push(Func {
                        params,
                        results: sig.results.clone(),
                        blocks: vec![block],
                    });
                    // Rewrite the call site to invoke the wrapper: prepend the handle to the op args.
                    let mut call_args = Vec::with_capacity(1 + args.len());
                    call_args.push(*handle);
                    call_args.extend(args.iter().copied());
                    *inst = Inst::Call {
                        func: g,
                        args: call_args,
                    };
                } else if let Inst::CallImport {
                    import,
                    sig,
                    handle,
                    args,
                } = inst
                {
                    let g = base + wrappers.len() as u32;
                    // Same wrapper shape as `cap.call`: (handle: i32, ...sig.params) -> sig.results.
                    // The import index is an immediate, so it stays baked into the wrapper body; the
                    // (vestigial) handle operand is threaded through like `cap.call`'s live one.
                    let mut params = Vec::with_capacity(1 + sig.params.len());
                    params.push(ValType::I32);
                    params.extend(sig.params.iter().copied());
                    let nparams = params.len() as u32;
                    let wrapper_args: Vec<u32> = (1..nparams).collect();
                    let ret: Vec<u32> = (nparams..nparams + sig.results.len() as u32).collect();
                    let block = Block {
                        params: params.clone(),
                        insts: vec![Inst::CallImport {
                            import: *import,
                            sig: sig.clone(),
                            handle: 0,
                            args: wrapper_args,
                        }],
                        term: Terminator::Return(ret),
                    };
                    wrappers.push(Func {
                        params,
                        results: sig.results.clone(),
                        blocks: vec![block],
                    });
                    let mut call_args = Vec::with_capacity(1 + args.len());
                    call_args.push(*handle);
                    call_args.extend(args.iter().copied());
                    *inst = Inst::Call {
                        func: g,
                        args: call_args,
                    };
                } else if let Inst::CapSelfResolve { name_ptr, name_len } = inst {
                    let g = base + wrappers.len() as u32;
                    // Fixed signature `(name_ptr: i64, name_len: i64) -> i32` (result appended at val 2).
                    let (np, nl) = (*name_ptr, *name_len);
                    let block = Block {
                        params: vec![ValType::I64, ValType::I64],
                        insts: vec![Inst::CapSelfResolve {
                            name_ptr: 0,
                            name_len: 1,
                        }],
                        term: Terminator::Return(vec![2]),
                    };
                    wrappers.push(Func {
                        params: vec![ValType::I64, ValType::I64],
                        results: vec![ValType::I32],
                        blocks: vec![block],
                    });
                    *inst = Inst::Call {
                        func: g,
                        args: vec![np, nl],
                    };
                }
            }
        }
    }
    m.funcs.extend(wrappers);
}

/// Compile a **whole-module reactor** guest with **widened cross-tier calls** (Doom-perf): emit every
/// reachable in-subset function to wasm and route a **direct** `Call` to any reachable, non-emitted,
/// **integer-signature** function through `env.call_interp` — not just the strict memory-free/call-free
/// [`interp_leaf`]s [`compile_module_mixed_entry`] allows. A cross-tier callee here may touch memory,
/// call other functions, and use capabilities, so the host's `call_interp` callback **must run it over
/// the SAME (shared) window + host** as the emitted code (a fresh window would lose its memory
/// effects) — the contract this mode adds over the leaf-only modes, which run leaves over a throwaway
/// window. This is what lets Doom's hot render path emit while its cold range-check / I/O helpers
/// (which make capability calls) stay on the interpreter.
///
/// A `call_indirect` may dispatch to a cross-tier target: an address-taken (`RefFunc`) function that
/// isn't emitted gets an identity-table slot holding a **trampoline** (a wasm function with the call
/// site's env-prepended signature that bounces to `env.call_interp`), so the indirect call reaches the
/// interpreter over the shared window just like a direct cross-tier call.
///
/// Returns the wasm plus a per-function **emitted** bitmap (`emitted[i]` ⇒ `f{i}` runs on wasm; the
/// rest are cross-tier). [`Error::Unsupported`] if the entry isn't in-subset, a reachable function has
/// a non-integer signature (can't be marshalled cross-tier), or an address-taken indirect target is
/// itself non-integer-signature (can't be trampolined).
pub fn compile_module_reactor(
    m: &Module,
    entry: u32,
    shared_memory: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    let n = m.funcs.len();
    let a = analyze_from(m, entry);
    // Cross-tier: reachable, not emitted (not in-subset), and marshallable (integer signature). Runs on
    // the interpreter over the shared window — so, unlike `interp_leaf`, memory/calls/caps are fine.
    let cross: Vec<bool> = (0..n)
        .map(|i| a.reachable[i] && !a.in_subset[i] && int_sig(&m.funcs[i]))
        .collect();
    let ok = (entry as usize) < n
        && a.in_subset[entry as usize]
        // Every reachable function must be emittable or cross-tier-callable. When the guest makes an
        // indirect call, `analyze` marks **all** functions reachable, so this also guarantees every
        // possible indirect target (including a data-segment function pointer, which no `RefFunc` scan
        // sees) is either emitted or `cross` — hence gets an identity-table slot: the emitted wasm
        // function, or a trampoline that bounces to the interpreter (see `emit_module`).
        && (0..n).all(|i| !a.reachable[i] || a.in_subset[i] || cross[i]);
    if !ok {
        return Err(Error::Unsupported("guest not cross-tier reactor runnable"));
    }
    let mut wasm_of: Vec<Option<u32>> = vec![None; n];
    let mut emitted: Vec<usize> = Vec::new();
    let mut emitted_bitmap = vec![false; n];
    for i in 0..n {
        if a.reachable[i] && a.in_subset[i] {
            wasm_of[i] = Some(IMPORTED_FUNCS + emitted.len() as u32);
            emitted.push(i);
            emitted_bitmap[i] = true;
        }
    }
    let wasm = emit_module(m, shared_memory, &emitted, &wasm_of, &cross)?;
    Ok((wasm, emitted_bitmap))
}

/// Compile a **tier-up** module for the browser threads tier (`BROWSER.md` § "wasm-JIT tier",
/// per-Worker JIT). Unlike [`compile_module_mixed_entry`], eligibility is **not** rooted at one
/// entry: the guest keeps running on the resumable interpreter (which drives `thread.spawn`/`join`,
/// atomics, `memory.wait`), and a direct `Call` to any emitted function surfaces as a *tier-up* the
/// host runs on the emitted region — so a pure compute leaf reachable **only** through
/// `thread.spawn` still emits, even though its caller (a concurrency orchestrator) never JITs.
///
/// Returns the emitted wasm plus the per-function eligibility bitmap: `eligible[i]` ⇒ `f{i}` is
/// exported and safe for the host to call. A function is emitted iff it is in-subset, and every
/// direct callee is itself emitted or a cross-tier interp leaf — a monotone fixpoint (start from
/// "every in-subset function", drop any whose emitted body would carry an unroutable `Call`). A
/// function that uses `call_indirect` is emitted only when the **whole** module is in-subset (so
/// every identity-table slot resolves to an emitted target); otherwise it is dropped, keeping the
/// emitted module table-free. [`Error::Unsupported`] only if the assembler itself rejects the set
/// (it never should, by construction) — an empty eligible set is a success with no `f{i}` exports.
pub fn compile_module_tierup(
    m: &Module,
    shared_memory: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    let n = m.funcs.len();
    let atomics_ok = module_atomics_ok(m);
    let in_subset: Vec<bool> = m
        .funcs
        .iter()
        .map(|f| func_in_subset(m, f, atomics_ok))
        .collect();
    let leaf: Vec<bool> = (0..n)
        .map(|i| !in_subset[i] && interp_leaf(&m.funcs[i]))
        .collect();
    let all_in_subset = in_subset.iter().all(|&s| s);

    // Optimistic start: every in-subset function is a candidate. A `call_indirect` can dispatch to any
    // identity-table slot, so a function that uses one is only safe to emit when every function is
    // in-subset (all slots resolve); otherwise drop it (and the emitted module needs no table).
    let mut emit: Vec<bool> = (0..n)
        .map(|i| in_subset[i] && (all_in_subset || !func_uses_indirect(&m.funcs[i])))
        .collect();
    // Fixpoint: drop any candidate that directly calls a function which is neither still a candidate
    // nor a cross-tier leaf — its emitted body would have an unroutable `Call`. Monotone (only
    // removes), so it converges in ≤ n passes.
    loop {
        let mut changed = false;
        for i in 0..n {
            if !emit[i] {
                continue;
            }
            for c in func_callees(&m.funcs[i]) {
                let c = c as usize;
                if c >= n || (!emit[c] && !leaf[c]) {
                    emit[i] = false;
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut wasm_of: Vec<Option<u32>> = vec![None; n];
    let mut emitted: Vec<usize> = Vec::new();
    for (i, e) in emit.iter().enumerate() {
        if *e {
            wasm_of[i] = Some(IMPORTED_FUNCS + emitted.len() as u32);
            emitted.push(i);
        }
    }
    let wasm = emit_module(m, shared_memory, &emitted, &wasm_of, &leaf)?;
    Ok((wasm, emit))
}

/// Assemble the wasm module: emit the functions listed in `emitted` (SVM indices, in the order they
/// take wasm indices), routing each `Call` via `wasm_of` (a direct wasm call) or, for an interp
/// leaf, through `env.call_interp`. See the module docs for the emitted shape.
fn emit_module(
    m: &Module,
    shared_memory: bool,
    emitted: &[usize],
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
) -> Result<Vec<u8>, Error> {
    // An import *manifest* is fine (IMPORTS.md phase 3): executable `call.import`s dispatch on the
    // import-capable interpreter tier, reached through outlined wrappers / cross-tier calls. What
    // must not happen is an import op surviving in a function this emitter actually lowers — the
    // tierability classifier excludes them, so this is a belt-and-braces check, not a filter.
    for &i in emitted {
        for b in &m.funcs[i].blocks {
            for inst in &b.insts {
                if matches!(inst, Inst::CallImport { .. } | Inst::ImportAttach { .. }) {
                    return Err(Error::Unsupported("import op in an emitted function"));
                }
            }
        }
    }
    // `data` segments are *not* rejected: the emitted code only loads/stores, so the **host** must
    // materialize the module's data into the window before the run (as the interpreter's window
    // init does) — the browser/bench linkers write `m.data` into the window first. An unwritten
    // segment simply reads as zero (and any resulting divergence is caught by the bench's
    // result-vs-native cross-check). Read-only enforcement (D40) is deferred with the §13 page ops.
    let mapped: u64 = match &m.memory {
        Some(mc) => 1u64 << mc.size_log2,
        None => 0,
    };

    // Types: 0 = env.trap `(i32) -> ()`, 1 = env.call_interp `(i32 func, i32 args_ptr) -> ()`, then
    // one per emitted function (dedup'd).
    let mut types: Vec<(Vec<u8>, Vec<u8>)> = vec![(vec![0x7f], vec![]), (vec![0x7f, 0x7f], vec![])];
    let mut fn_type_idx: Vec<u32> = Vec::with_capacity(emitted.len());
    for &fi in emitted {
        let f = &m.funcs[fi];
        let mut params = vec![0x7f, 0x7f]; // win: i32, env: i32
        for p in &f.params {
            params.push(valtype_byte(*p)?);
        }
        let mut results = Vec::with_capacity(f.results.len());
        for r in &f.results {
            results.push(valtype_byte(*r)?);
        }
        let ty = (params, results);
        let idx = match types.iter().position(|t| *t == ty) {
            Some(i) => i,
            None => {
                types.push(ty);
                types.len() - 1
            }
        };
        fn_type_idx.push(idx as u32);
    }

    // `call_indirect` needs its (prepended-env) signature declared in the type section too; add any
    // not already present, and note whether the module needs a funcref table + element segment.
    let mut needs_table = false;
    for &fi in emitted {
        for b in &m.funcs[fi].blocks {
            // A `call_indirect` type shows up as an instruction; a `return_call_indirect` as the
            // block terminator — both dispatch through the table and need their signature declared.
            let indirect_ty = b
                .insts
                .iter()
                .filter_map(|inst| match inst {
                    Inst::CallIndirect { ty, .. } => Some(ty),
                    _ => None,
                })
                .chain(match &b.term {
                    Terminator::ReturnCallIndirect { ty, .. } => Some(ty),
                    _ => None,
                });
            for ty in indirect_ty {
                needs_table = true;
                let key = indirect_type_bytes(ty)?;
                if !types.contains(&key) {
                    types.push(key);
                }
            }
        }
    }
    // The identity funcref table (`RefFunc`/`dispatch_indirect` semantics): slot `s` = SVM function
    // `s`, power-of-two length, trapping (null) padding — matching the interpreter's `DomainTable`
    // (`funcs.len().next_power_of_two()`, `reserve_log2 = 0`). Masking `idx & (table_size - 1)` in
    // the lowering reproduces `dispatch_indirect`'s `idx & (len - 1)`.
    let table_size = m.funcs.len().next_power_of_two().max(1) as u32;

    // Cross-tier indirect trampolines. A function whose address is taken (`RefFunc`) but which is
    // *not* emitted still occupies an identity-table slot; an indirect call to it must reach the
    // interpreter. For each such **cross-tier** (`interp_leaf`) address-taken function we emit a
    // standalone trampoline — a wasm function with the same env-prepended `call_indirect` signature
    // that does `env.call_interp` (see [`emit_trampoline`]). Every remaining non-emitted slot gets a
    // `()->()` trap stub, so a forged/mistyped index fails closed at the signature check. Trampolines
    // and the trap stub take wasm indices *after* the emitted functions (imports + emitted + these).
    let mut tramp_of: Vec<Option<u32>> = vec![None; m.funcs.len()];
    let mut extra_type_idx: Vec<u32> = Vec::new();
    let mut extra_bodies: Vec<Vec<u8>> = Vec::new();
    let mut trap_stub_widx: Option<u32> = None;
    if needs_table {
        // **Every** cross-tier function needs a trampoline slot — not just the `RefFunc`
        // address-taken ones. A function pointer can be an indirect-call target without any `RefFunc`
        // instruction: the frontend bakes static function-pointer tables (e.g. Doom's `states[]` /
        // `mobjinfo[]` action functions) into **data segments** as plain function-index constants,
        // invisible to a RefFunc scan. So the identity table must route *any* index to its function,
        // exactly as the interpreter's `DomainTable` does — otherwise a `call_indirect` through a
        // data-segment pointer hits a trap stub ("null function or function signature mismatch") the
        // interpreter would have dispatched. (Fixed a hang/trap ~frame 174 of Doom, when the first
        // monster thinker fires an `A_*` action loaded from `states[]`.)
        let mut next_widx = IMPORTED_FUNCS + emitted.len() as u32;
        for fi in 0..m.funcs.len() {
            if wasm_of[fi].is_none() && interp_leaf[fi] {
                let f = &m.funcs[fi];
                let key = indirect_type_bytes(&FuncType {
                    params: f.params.clone(),
                    results: f.results.clone(),
                })?;
                let ti = match types.iter().position(|t| *t == key) {
                    Some(i) => i as u32,
                    None => {
                        types.push(key);
                        (types.len() - 1) as u32
                    }
                };
                extra_type_idx.push(ti);
                extra_bodies.push(emit_trampoline(f, fi as u32)?);
                tramp_of[fi] = Some(next_widx);
                next_widx += 1;
            }
        }
        // Any non-emitted, non-trampoline real slot needs the trap stub (`()->()`).
        let need_stub =
            (0..m.funcs.len()).any(|fi| wasm_of[fi].is_none() && tramp_of[fi].is_none());
        if need_stub {
            let key = (Vec::new(), Vec::new());
            let ti = match types.iter().position(|t| *t == key) {
                Some(i) => i as u32,
                None => {
                    types.push(key);
                    (types.len() - 1) as u32
                }
            };
            extra_type_idx.push(ti);
            extra_bodies.push(emit_trap_stub());
            trap_stub_widx = Some(next_widx);
        }
    }

    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(emitted.len() + extra_bodies.len());
    for &fi in emitted {
        bodies.push(emit_func(
            m,
            &m.funcs[fi],
            mapped,
            wasm_of,
            interp_leaf,
            &types,
            table_size,
        )?);
    }
    bodies.extend(extra_bodies);

    // ---- assemble the module ----
    let mut out = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // \0asm v1

    let mut sec = Vec::new(); // type section (1)
    uleb(&mut sec, types.len() as u64);
    for (params, results) in &types {
        sec.push(0x60);
        uleb(&mut sec, params.len() as u64);
        sec.extend_from_slice(params);
        uleb(&mut sec, results.len() as u64);
        sec.extend_from_slice(results);
    }
    section(&mut out, 1, &sec);

    let mut sec = Vec::new(); // import section (2): env.memory, env.trap (type 0), env.call_interp (type 1)
    uleb(&mut sec, 3);
    import_name(&mut sec, "env", "memory");
    sec.push(0x02); // memory
    if shared_memory {
        // Flag 0x03 = shared + has-max; min 0, max 65536 (wasm32's 4 GiB / 64 KiB-page ceiling). A
        // min-0/max-ceiling import is satisfied by any shared memory the host provides, and the
        // shared flag must match the provided memory's (the browser threads build's shared memory).
        sec.push(0x03);
        uleb(&mut sec, 0);
        uleb(&mut sec, 65536);
    } else {
        // Flag 0x00 = min-only, non-shared (the `wasmi` differential + a plain cdylib build).
        sec.push(0x00);
        uleb(&mut sec, 0);
    }
    import_name(&mut sec, "env", "trap");
    sec.push(0x00); // func
    uleb(&mut sec, 0); // type index 0
    import_name(&mut sec, "env", "call_interp");
    sec.push(0x00); // func
    uleb(&mut sec, 1); // type index 1
    section(&mut out, 2, &sec);

    let mut sec = Vec::new(); // function section (3): emitted, then trampolines + trap stub
    uleb(&mut sec, (fn_type_idx.len() + extra_type_idx.len()) as u64);
    for ti in fn_type_idx.iter().chain(&extra_type_idx) {
        uleb(&mut sec, *ti as u64);
    }
    section(&mut out, 3, &sec);

    if needs_table {
        let mut sec = Vec::new(); // table section (4): one funcref table, min = table_size
        uleb(&mut sec, 1);
        sec.push(0x70); // funcref elemtype
        sec.push(0x00); // limits flag 0x00 = min only
        uleb(&mut sec, table_size as u64);
        section(&mut out, 4, &sec);
    }

    let mut sec = Vec::new(); // export section (7): "f{svm_idx}" → its wasm index
    uleb(&mut sec, emitted.len() as u64);
    for &fi in emitted {
        let name = format!("f{fi}");
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x00);
        uleb(&mut sec, wasm_of[fi].unwrap() as u64);
    }
    section(&mut out, 7, &sec);

    if needs_table {
        // Element section (9): one active segment filling the identity table `[0, funcs.len())`.
        // Each real slot resolves to the function's wasm index: an emitted function (`wasm_of`), a
        // cross-tier **trampoline** (`tramp_of`, an address-taken interp-leaf), or the `()->()` trap
        // stub (unreachable / non-address-taken cross-tier functions — never a legitimate indirect
        // target, so their slot fails closed at the call_indirect signature check). Padding slots
        // `[funcs.len(), table_size)` stay null (they trap like the interpreter's `TABLE_EMPTY`).
        let mut segment = Vec::new();
        for fi in 0..m.funcs.len() {
            let widx = wasm_of[fi]
                .or(tramp_of[fi])
                .or(trap_stub_widx)
                .ok_or(Error::Unsupported("indirect call target not routable"))?;
            uleb(&mut segment, widx as u64);
        }
        let mut sec = Vec::new();
        uleb(&mut sec, 1); // one segment
        sec.push(0x00); // flags 0: active, table 0, i32 offset expr, funcidx vec
        sec.push(OP_I32_CONST); // offset expr: i32.const 0; end
        sleb32(&mut sec, 0);
        sec.push(OP_END);
        uleb(&mut sec, m.funcs.len() as u64);
        sec.extend_from_slice(&segment);
        section(&mut out, 9, &sec);
    }

    let mut sec = Vec::new(); // code section (10)
    uleb(&mut sec, bodies.len() as u64);
    for b in &bodies {
        uleb(&mut sec, b.len() as u64);
        sec.extend_from_slice(b);
    }
    section(&mut out, 10, &sec);

    Ok(out)
}

/// Emit a **cross-tier indirect trampoline** body for SVM function `fi` (its `Func` is `f`): a wasm
/// function with the env-prepended signature `(win:i32, env:i32, ...params) -> results` that marshals
/// its params into the env scratch, calls `env.call_interp(fi, args_ptr)`, and returns the result
/// slots — the same sequence [`emit_func`] uses for a cross-tier *direct* call, packaged as a
/// standalone function so a cross-tier function whose **address is taken** can fill its funcref-table
/// slot (an indirect call to it then reaches the interpreter). No locals: params are locals
/// `2..2+nparams`; results are loaded straight onto the operand stack for the return.
fn emit_trampoline(f: &Func, fi: u32) -> Result<Vec<u8>, Error> {
    if f.params.len().max(f.results.len()) > XCALL_MAX_SLOTS {
        return Err(Error::Unsupported("indirect trampoline arity too large"));
    }
    let mut code = Vec::new();
    uleb(&mut code, 0); // no local declarations
    for (i, p) in f.params.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 1); // env
        code.push(OP_I32_CONST);
        sleb32(&mut code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
        code.push(0x6a); // i32.add → slot addr
        code.push(OP_LOCAL_GET);
        uleb(&mut code, (2 + i) as u64); // the i-th SVM param local
        if *p == ValType::I32 {
            code.push(0xad); // i64.extend_i32_u
        }
        code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8
    }
    code.push(OP_I32_CONST);
    sleb32(&mut code, fi as i32);
    code.push(OP_LOCAL_GET);
    uleb(&mut code, 1); // env
    code.push(OP_I32_CONST);
    sleb32(&mut code, ENV_SCRATCH_OFF as i32);
    code.push(0x6a); // i32.add → args_ptr
    code.push(OP_CALL);
    uleb(&mut code, 1); // env.call_interp
    for (i, r) in f.results.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 1); // env
        code.push(OP_I32_CONST);
        sleb32(&mut code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
        code.push(0x6a); // i32.add
        code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8
        if *r == ValType::I32 {
            code.push(0xa7); // i32.wrap_i64
        }
    }
    code.push(OP_END);
    Ok(code)
}

/// A `() -> ()` **trap stub** body (`unreachable`). Fills funcref-table slots for functions that are
/// neither emitted nor a cross-tier trampoline (unreachable / non-address-taken cross-tier functions):
/// a verified guest only forms a funcref via `RefFunc` (an address-taken function), so such a slot is
/// never legitimately reached; if a forged/mistyped index hits it, `call_indirect`'s type check traps
/// (the stub's `()->()` type never matches a real `(win,env,…)` call site) — fail-closed, matching the
/// interpreter's `IndirectCallType`/`TABLE_EMPTY` trap.
fn emit_trap_stub() -> Vec<u8> {
    let mut code = Vec::new();
    uleb(&mut code, 0); // no locals
    code.push(0x00); // unreachable
    code.push(OP_END);
    code
}

/// The wasm function-type of a `call_indirect` signature: the two prepended env params (`win`,
/// `env`) ahead of the SVM param/result types — identical in shape to how [`emit_module`] types the
/// emitted functions, so wasm's built-in `call_indirect` signature check **is** the §3c type-id
/// check (a mismatch traps, exactly like `dispatch_indirect`'s `IndirectCallType`).
fn indirect_type_bytes(ty: &FuncType) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut params = vec![0x7f, 0x7f]; // win: i32, env: i32
    for p in &ty.params {
        params.push(valtype_byte(*p)?);
    }
    let mut results = Vec::with_capacity(ty.results.len());
    for r in &ty.results {
        results.push(valtype_byte(*r)?);
    }
    Ok((params, results))
}

/// The type-section index of a `call_indirect` signature (pre-added to `types` by [`emit_module`]).
fn indirect_type_index(types: &[(Vec<u8>, Vec<u8>)], ty: &FuncType) -> Result<u32, Error> {
    let key = indirect_type_bytes(ty)?;
    types
        .iter()
        .position(|t| *t == key)
        .map(|i| i as u32)
        .ok_or(Error::Unsupported("indirect call type not declared"))
}

fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    uleb(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn import_name(out: &mut Vec<u8>, module: &str, name: &str) {
    uleb(out, module.len() as u64);
    out.extend_from_slice(module.as_bytes());
    uleb(out, name.len() as u64);
    out.extend_from_slice(name.as_bytes());
}

/// Per-function emission state: the (block, value) → wasm-local map plus the scratch locals.
struct FnCtx {
    /// `local_of[block][value]` — wasm local index of each block-scoped SSA value.
    local_of: Vec<Vec<u32>>,
    next_l: u32,
    ea_l: u32,
    fuel_l: u32,
    /// i32 scratch holding a confined atomic address so a read-modify-write / compare-exchange can
    /// reuse it for both the load and the store without recomputing (and re-confining) it.
    atomic_addr_l: u32,
    /// Open label count inside the body; the dispatcher `loop` is the first label opened, so a
    /// branch back to it from depth `d` is `br (d - 1)`.
    depth: u32,
}

impl FnCtx {
    fn br_dispatch(&self, code: &mut Vec<u8>) {
        code.push(OP_BR);
        uleb(code, (self.depth - 1) as u64);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_func(
    m: &Module,
    f: &Func,
    mapped: u64,
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
    types: &[(Vec<u8>, Vec<u8>)],
    table_size: u32,
) -> Result<Vec<u8>, Error> {
    let n_params = 2 + f.params.len() as u32; // win, env, then the SVM params

    // Allocate locals with a **per-type pool reused across blocks**, then $next/$ea/$fuel/$atomic.
    //
    // Values are block-scoped (SSA numbering resets per block; all cross-block dataflow goes through
    // block params), and the dispatcher runs exactly one block at a time — so when block B executes,
    // every other block's value locals are dead. Their slots can therefore be shared: a type needs
    // only `max over blocks of (that type's value count in the block)` locals, not the sum. This is
    // what keeps a huge function (QuickJS's ~1800-block `JS_CallInternal`) under wasm engines'
    // per-function local cap — the sum would be hundreds of thousands, the max is a few thousand.
    //
    // Sharing is safe because the only cross-block local write, `emit_edge`, pushes **all** branch
    // args onto the operand stack before storing any target param, so a target param slot that aliases
    // a source value slot still reads the old value first (the same property that already made a
    // param-permuting self-branch safe). Within a block each value keeps a distinct slot (assigned by
    // per-type rank), so no live value is clobbered.
    let per_block_types: Vec<Vec<ValType>> = f
        .blocks
        .iter()
        .map(|b| block_value_types(m, b))
        .collect::<Result<_, _>>()?;
    // Pool size per type = the max count of that type in any single block.
    const NTYPES: usize = 6; // I32, I64, F32, F64, V128, Ref (the ValType variants)
    let type_slot = |t: ValType| -> usize {
        match t {
            ValType::I32 => 0,
            ValType::I64 => 1,
            ValType::F32 => 2,
            ValType::F64 => 3,
            ValType::V128 => 4,
            ValType::Ref => 5,
        }
    };
    let mut pool: [u32; NTYPES] = [0; NTYPES];
    for tys in &per_block_types {
        let mut per_block = [0u32; NTYPES];
        for t in tys {
            per_block[type_slot(*t)] += 1;
        }
        for i in 0..NTYPES {
            pool[i] = pool[i].max(per_block[i]);
        }
    }
    // Lay the pools out contiguously; `base[t]` is the first local index (past the wasm params) of
    // type `t`'s pool.
    let mut base = [0u32; NTYPES];
    let mut acc = 0u32;
    for i in 0..NTYPES {
        base[i] = acc;
        acc += pool[i];
    }
    let mut local_types: Vec<ValType> = Vec::with_capacity(acc as usize + 4);
    for (i, &t) in [
        ValType::I32,
        ValType::I64,
        ValType::F32,
        ValType::F64,
        ValType::V128,
        ValType::Ref,
    ]
    .iter()
    .enumerate()
    {
        for _ in 0..pool[i] {
            local_types.push(t);
        }
    }
    // Map each block's values to pool slots: value `v` of type `t` gets `base[t] + (its rank among
    // same-typed values in the block)`. Reused across blocks — block B's slots overlap block A's.
    let local_of: Vec<Vec<u32>> = per_block_types
        .iter()
        .map(|tys| {
            let mut used = [0u32; NTYPES];
            tys.iter()
                .map(|t| {
                    let s = type_slot(*t);
                    let idx = n_params + base[s] + used[s];
                    used[s] += 1;
                    idx
                })
                .collect()
        })
        .collect();
    let next_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I32);
    let ea_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I64);
    let fuel_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I64);
    let atomic_addr_l = n_params + local_types.len() as u32;
    local_types.push(ValType::I32);

    let mut cx = FnCtx {
        local_of,
        next_l,
        ea_l,
        fuel_l,
        atomic_addr_l,
        depth: 0,
    };

    let mut code = Vec::new();
    // Copy the SVM params into the entry block's param locals ($next defaults to 0 = entry).
    for (i, _) in f.params.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 2 + i as u64);
        code.push(OP_LOCAL_SET);
        uleb(&mut code, cx.local_of[0][i] as u64);
    }

    // The dispatcher: loop { fuel; block .. block { br_table $next } code_0 .. code_{N-1} }.
    code.push(OP_LOOP);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_fuel_check(&mut cx, &mut code);
    let n = f.blocks.len();
    for _ in 0..n {
        code.push(OP_BLOCK);
        code.push(BLOCKTYPE_VOID);
        cx.depth += 1;
    }
    code.push(OP_LOCAL_GET);
    uleb(&mut code, cx.next_l as u64);
    code.push(OP_BR_TABLE);
    uleb(&mut code, n as u64); // n labels + default
    for k in 0..n {
        uleb(&mut code, k as u64); // depth k exits block k → lands at code_k
    }
    uleb(&mut code, (n - 1) as u64); // default: unreachable by construction; any valid label

    for (k, b) in f.blocks.iter().enumerate() {
        code.push(OP_END); // close block k; code_k follows
        cx.depth -= 1;
        emit_block_body(
            m,
            f,
            &mut cx,
            &mut code,
            k,
            b,
            &per_block_types[k],
            mapped,
            wasm_of,
            interp_leaf,
            types,
            table_size,
        )?;
    }
    code.push(OP_END); // close the loop
    cx.depth -= 1;
    code.push(OP_UNREACHABLE); // every path returned / trapped / re-dispatched
    code.push(OP_END); // function body end

    // Prepend the locals vector (grouped runs of one type).
    let mut body = Vec::new();
    let mut groups: Vec<(u32, u8)> = Vec::new();
    for t in &local_types {
        let byte = valtype_byte(*t)?;
        match groups.last_mut() {
            Some((count, b)) if *b == byte => *count += 1,
            _ => groups.push((1, byte)),
        }
    }
    uleb(&mut body, groups.len() as u64);
    for (count, byte) in groups {
        uleb(&mut body, count as u64);
        body.push(byte);
    }
    body.extend_from_slice(&code);
    Ok(body)
}

/// Debit one fuel unit from the `i64` cell at `env` and trap `TRAP_OUT_OF_FUEL` when it goes
/// negative. Runs once per dispatcher iteration — a coarser debit than the interpreter's per-op
/// fuel, deliberately (fuel is a §5 bound, not an observable).
fn emit_fuel_check(cx: &mut FnCtx, code: &mut Vec<u8>) {
    code.push(OP_LOCAL_GET); // [env]        (store address)
    uleb(code, 1);
    code.push(OP_LOCAL_GET); // [env, env]
    uleb(code, 1);
    code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8 → [env, fuel]
    code.push(OP_I64_CONST);
    sleb64(code, 1);
    code.push(0x7d); // i64.sub → [env, fuel-1]
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.fuel_l as u64);
    code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8 → []
    code.push(OP_LOCAL_GET);
    uleb(code, cx.fuel_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, 0);
    code.push(0x53); // i64.lt_s
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_OUT_OF_FUEL);
    code.push(OP_END);
    cx.depth -= 1;
}

/// `call env.trap(code); unreachable` — the host records the SVM trap kind, the `unreachable`
/// aborts execution.
fn emit_trap(code: &mut Vec<u8>, trap_code: i32) {
    code.push(OP_I32_CONST);
    sleb32(code, trap_code);
    code.push(OP_CALL);
    uleb(code, 0); // func 0 = the env.trap import
    code.push(OP_UNREACHABLE);
}

/// Confine the effective address for a `width`-byte access under **trap-confinement** (§4, D38),
/// leaving the confined 32-bit linear-memory address on the stack: bounds-check the *unmasked*
/// `eff = addr + offset` (trap `MemoryFault` unless `eff <= mapped - width` — exactly the
/// trap-confinement `svm_mask::Window::checked`, so an out-of-window address faults instead of
/// wrapping back in), then compute `win + (eff & MASK)`. The `& MASK` clamp is a no-op past the
/// check (`eff < mapped ≤ reserved`), kept to mirror the native JIT's check+clamp lowering and to
/// keep the following `i32.wrap` in-window as defense-in-depth.
fn emit_confine(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    addr_local: u32,
    offset: u64,
    width: u64,
    mapped: u64,
) {
    emit_confine_maybe_aligned(cx, code, addr_local, offset, width, mapped, false)
}

/// Like [`emit_confine`] but, when `align`, also traps `MemoryFault` on a **misaligned** effective
/// address (`eff % width != 0`) — the natural-alignment requirement §12 atomics carry (the
/// interpreter's `check_align`), which a real hardware atomic would also raise. `width` is a power of
/// two for the atomic types (4 or 8), so `width - 1` is the alignment mask.
fn emit_confine_maybe_aligned(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    addr_local: u32,
    offset: u64,
    width: u64,
    mapped: u64,
    align: bool,
) {
    code.push(OP_LOCAL_GET);
    uleb(code, addr_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, offset as i64);
    code.push(0x7c); // i64.add → eff (unmasked)
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, mapped.wrapping_sub(width) as i64);
    code.push(0x56); // i64.gt_u: eff > mapped - width ?
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    if align {
        // `eff & (width - 1) != 0` ⇒ misaligned ⇒ trap (matches `check_align`).
        code.push(OP_LOCAL_GET);
        uleb(code, cx.ea_l as u64);
        code.push(OP_I64_CONST);
        sleb64(code, (width - 1) as i64);
        code.push(0x83); // i64.and
        code.push(OP_I64_CONST);
        sleb64(code, 0);
        code.push(0x52); // i64.ne → misaligned?
        code.push(OP_IF);
        code.push(BLOCKTYPE_VOID);
        cx.depth += 1;
        emit_trap(code, TRAP_MEMORY_FAULT);
        code.push(OP_END);
        cx.depth -= 1;
    }
    code.push(OP_LOCAL_GET);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and → clamp (no-op past the check)
    code.push(0xa7); // i32.wrap_i64
    code.push(OP_LOCAL_GET);
    uleb(code, 0); // win
    code.push(0x6a); // i32.add → the confined linear-memory address
}

/// Open `if len != 0 {` for a bulk op — the caller emits the confined op inside and closes with a
/// matching `OP_END` (`cx.depth -= 1`). A zero-length bulk op is a no-op that must never fault (see
/// the lowering comment), so the entire span-check + `memory.fill`/`.copy` lives under this guard.
fn emit_bulk_guard_open(cx: &mut FnCtx, code: &mut Vec<u8>, len_local: u32) {
    code.push(OP_LOCAL_GET);
    uleb(code, len_local as u64);
    code.push(0x50); // i64.eqz → (len == 0)
    code.push(0x45); // i32.eqz → (len != 0)
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
}

/// **Whole-span confinement** for a bulk op (`memory.copy`/`memory.fill`) — the `len`-is-a-value
/// analogue of [`emit_confine`], and the security hinge for D62 bulk memory. Traps `MemoryFault` unless
/// the span `[base, base+len)` lies within `[0, mapped)` — matching the interpreter's `confine_span` +
/// `check_prot_span` net behaviour over a fresh window (a span above `mapped` is uncommitted → faults),
/// and keeping every accessed byte inside the physical window (never the adjacent linear memory). The
/// check is **overflow-safe**: `base > mapped` then `len > mapped - base` (the second computed only
/// once `base <= mapped`, so `mapped - base` can't underflow and `base + len` can't overflow).
///
/// Called **inside an `if len != 0` guard** (see the lowering in `emit_block_body`), so `len >= 1`
/// here and a passed check guarantees `base < mapped` — which makes [`emit_win_addr`]'s mask a no-op.
/// Emits nothing to the operand stack; call [`emit_win_addr`] afterwards for each span's confined
/// address.
fn emit_span_check(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    base_local: u32,
    len_local: u32,
    mapped: u64,
) {
    // trap if base > mapped
    code.push(OP_LOCAL_GET);
    uleb(code, base_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, mapped as i64);
    code.push(0x56); // i64.gt_u
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    // trap if len > mapped - base
    code.push(OP_LOCAL_GET);
    uleb(code, len_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, mapped as i64);
    code.push(OP_LOCAL_GET);
    uleb(code, base_local as u64);
    code.push(0x7d); // i64.sub → mapped - base (base <= mapped here)
    code.push(0x56); // i64.gt_u: len > mapped - base
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
}

/// Push the confined linear-memory address `win + (base & MASK)` (an `i32`) for a bulk-op span whose
/// `base` local has already passed [`emit_span_check`] (so `base < mapped ≤ 2^32` and the `& MASK` is a
/// no-op clamp, mirroring the scalar path's defense-in-depth). `mapped ≤ 2^32` on wasm32, so the later
/// `i32.wrap` of a checked `len` is exact.
fn emit_win_addr(code: &mut Vec<u8>, base_local: u32) {
    code.push(OP_LOCAL_GET);
    uleb(code, base_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and → clamp into the window
    code.push(0xa7); // i32.wrap_i64
    code.push(OP_LOCAL_GET);
    uleb(code, 0); // win
    code.push(0x6a); // i32.add
}

/// Push branch args onto the operand stack, then pop them into the target block's param locals in
/// reverse — stack copies make a param-permuting self-branch safe.
fn emit_edge(
    cx: &FnCtx,
    code: &mut Vec<u8>,
    from_block: usize,
    target: u32,
    args: &[svm_ir::ValIdx],
) {
    for a in args {
        code.push(OP_LOCAL_GET);
        uleb(code, cx.local_of[from_block][*a as usize] as u64);
    }
    for i in (0..args.len()).rev() {
        code.push(OP_LOCAL_SET);
        uleb(code, cx.local_of[target as usize][i] as u64);
    }
    code.push(OP_I32_CONST);
    sleb32(code, target as i32);
    code.push(OP_LOCAL_SET);
    uleb(code, cx.next_l as u64);
}

#[allow(clippy::too_many_arguments)]
fn emit_block_body(
    m: &Module,
    f: &Func,
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    k: usize,
    b: &Block,
    value_types: &[ValType],
    mapped: u64,
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
    types: &[(Vec<u8>, Vec<u8>)],
    table_size: u32,
) -> Result<(), Error> {
    let mut next_val = b.params.len(); // where the next instruction's results land
    let get = |code: &mut Vec<u8>, cx: &FnCtx, v: svm_ir::ValIdx| {
        code.push(OP_LOCAL_GET);
        uleb(code, cx.local_of[k][v as usize] as u64);
    };
    for inst in &b.insts {
        match inst {
            Inst::ConstI32(v) => {
                code.push(OP_I32_CONST);
                sleb32(code, *v);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ConstI64(v) => {
                code.push(OP_I64_CONST);
                sleb64(code, *v);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntBin { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(intbin_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntCmp { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(intcmp_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IntUn { ty, op, a } => {
                get(code, cx, *a);
                code.push(intun_opcode(*ty, *op)?);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Eqz { ty, a } => {
                get(code, cx, *a);
                code.push(match ty {
                    IntTy::I32 => 0x45,
                    IntTy::I64 => 0x50,
                });
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Convert { op, a } => {
                get(code, cx, *a);
                code.push(match op {
                    ConvOp::ExtendI32S => 0xac,
                    ConvOp::ExtendI32U => 0xad,
                    ConvOp::WrapI64 => 0xa7,
                });
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Select { cond, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                get(code, cx, *cond);
                code.push(OP_SELECT);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Load {
                op, addr, offset, ..
            } => {
                let (opcode, width, _) = load_op(*op)?;
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                );
                code.extend_from_slice(&[opcode, 0x00, 0x00]); // align=1, offset=0
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let (opcode, width) = store_op(*op)?;
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                );
                get(code, cx, *value);
                code.extend_from_slice(&[opcode, 0x00, 0x00]); // align=1, offset=0
            }
            // ---- §12 atomics (single-threaded lowering; see the `block_value_types` note) ---------
            // Each confines + natural-align-traps the effective address, then runs the plain memory
            // op. For a JIT-tier (single-threaded) guest this is observably identical to a hardware
            // atomic, and stays differential-testable on `wasmi`.
            Inst::AtomicLoad {
                ty, addr, offset, ..
            } => {
                let (load, _store, width) = atomic_ops(*ty);
                emit_confine_maybe_aligned(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                    true,
                );
                code.extend_from_slice(&[load, 0x00, 0x00]);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::AtomicStore {
                ty,
                addr,
                value,
                offset,
                ..
            } => {
                let (_load, store, width) = atomic_ops(*ty);
                emit_confine_maybe_aligned(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                    true,
                );
                get(code, cx, *value);
                code.extend_from_slice(&[store, 0x00, 0x00]);
            }
            Inst::AtomicRmw {
                ty,
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let (load, store, width) = atomic_ops(*ty);
                let res = cx.local_of[k][next_val]; // holds the returned **old** value
                emit_confine_maybe_aligned(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                    true,
                );
                code.push(OP_LOCAL_SET);
                uleb(code, cx.atomic_addr_l as u64); // save the confined address
                code.push(OP_LOCAL_GET);
                uleb(code, cx.atomic_addr_l as u64);
                code.extend_from_slice(&[load, 0x00, 0x00]); // old = *addr
                code.push(OP_LOCAL_SET);
                uleb(code, res as u64); // res = old
                                        // *addr = op(old, value)  (xchg ignores old — store `value` directly)
                code.push(OP_LOCAL_GET);
                uleb(code, cx.atomic_addr_l as u64);
                match op {
                    AtomicRmwOp::Xchg => get(code, cx, *value),
                    _ => {
                        code.push(OP_LOCAL_GET);
                        uleb(code, res as u64); // old
                        get(code, cx, *value);
                        code.push(intbin_opcode(*ty, rmw_binop(*op)));
                    }
                }
                code.extend_from_slice(&[store, 0x00, 0x00]);
                next_val += 1; // res already holds the old value
            }
            Inst::AtomicCmpxchg {
                ty,
                addr,
                expected,
                replacement,
                offset,
                ..
            } => {
                let (load, store, width) = atomic_ops(*ty);
                let res = cx.local_of[k][next_val]; // holds the returned **old** value
                emit_confine_maybe_aligned(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    width,
                    mapped,
                    true,
                );
                code.push(OP_LOCAL_SET);
                uleb(code, cx.atomic_addr_l as u64);
                code.push(OP_LOCAL_GET);
                uleb(code, cx.atomic_addr_l as u64);
                code.extend_from_slice(&[load, 0x00, 0x00]); // old = *addr
                code.push(OP_LOCAL_SET);
                uleb(code, res as u64); // res = old
                                        // if old == expected { *addr = replacement }  (value width == type width, no mask)
                code.push(OP_LOCAL_GET);
                uleb(code, res as u64);
                get(code, cx, *expected);
                code.push(intcmp_opcode(*ty, CmpOp::Eq));
                code.push(OP_IF);
                code.push(BLOCKTYPE_VOID);
                cx.depth += 1;
                code.push(OP_LOCAL_GET);
                uleb(code, cx.atomic_addr_l as u64);
                get(code, cx, *replacement);
                code.extend_from_slice(&[store, 0x00, 0x00]);
                code.push(OP_END);
                cx.depth -= 1;
                next_val += 1; // res already holds the old value
            }
            // ---- bulk memory (D62): whole-span confinement, then `memory.fill`/`memory.copy` ----
            // The security hinge. The whole op runs under `if len != 0`, mirroring the interpreter's
            // `if len == 0 { return Ok }` short-circuit: a bulk op that touches no byte is an
            // unconditional no-op — it must NOT fault even at a wild base (and wasm's own
            // `memory.fill`/`.copy` would otherwise bounds-check the base *before* its `n == 0`
            // early-out, faulting a masked-but-out-of-linear-memory address). Inside the guard
            // `emit_span_check` traps unless the whole span is in `[0, mapped)`; then `emit_win_addr`
            // masks each base into the window (a no-op past the check) — same net confinement as the
            // per-byte `Store` path, proven once per span. `len` (i64) is `i32.wrap`ped after the
            // check (exact, since `len <= mapped <= 2^32`); `val` is already the i32 fill byte.
            Inst::MemFill { dst, val, len } => {
                let dl = cx.local_of[k][*dst as usize];
                let ll = cx.local_of[k][*len as usize];
                emit_bulk_guard_open(cx, code, ll);
                emit_span_check(cx, code, dl, ll, mapped);
                emit_win_addr(code, dl); // dest addr (i32)
                get(code, cx, *val); // fill byte (already i32)
                get(code, cx, *len);
                code.push(0xa7); // i32.wrap_i64 → size (i32)
                code.extend_from_slice(&[0xFC, 0x0B, 0x00]); // memory.fill mem=0
                code.push(OP_END); // close `if len != 0`
                cx.depth -= 1;
            }
            // `memory.copy` is overlap-safe, so it lowers both `MemCopy` (non-overlapping) and
            // `MemMove` (overlap-safe) — the stronger op is always a correct refinement.
            Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
                let dl = cx.local_of[k][*dst as usize];
                let sl = cx.local_of[k][*src as usize];
                let ll = cx.local_of[k][*len as usize];
                emit_bulk_guard_open(cx, code, ll);
                emit_span_check(cx, code, dl, ll, mapped);
                emit_span_check(cx, code, sl, ll, mapped);
                emit_win_addr(code, dl); // dest addr (i32)
                emit_win_addr(code, sl); // src addr (i32)
                get(code, cx, *len);
                code.push(0xa7); // i32.wrap_i64 → size (i32)
                code.extend_from_slice(&[0xFC, 0x0A, 0x00, 0x00]); // memory.copy dst=0 src=0
                code.push(OP_END); // close `if len != 0`
                cx.depth -= 1;
            }
            // ---- scalar floats (all 1:1 with core wasm) ----
            Inst::ConstF32(bits) => {
                code.push(0x43); // f32.const
                code.extend_from_slice(&bits.to_le_bytes());
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ConstF64(bits) => {
                code.push(0x44); // f64.const
                code.extend_from_slice(&bits.to_le_bytes());
                set_result(cx, code, k, &mut next_val);
            }
            Inst::FBin { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(fbin_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::FUn { ty, op, a } => {
                get(code, cx, *a);
                code.push(fun_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::FCmp { ty, op, a, b: rb } => {
                get(code, cx, *a);
                get(code, cx, *rb);
                code.push(fcmp_opcode(*ty, *op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::FToISat { op, a } => {
                get(code, cx, *a);
                code.push(0xfc); // saturating-truncation prefix
                code.push(ftoisat_subop(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::FToITrap { op, a } => {
                get(code, cx, *a);
                code.push(ftoitrap_opcode(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::IToFConv { op, a } => {
                get(code, cx, *a);
                code.push(itof_opcode(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Cast { op, a } => {
                get(code, cx, *a);
                code.push(cast_opcode(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Call { func, args } => {
                let callee = &m.funcs[*func as usize];
                let n_results = callee.results.len();
                match wasm_of[*func as usize] {
                    // Same-tier: a direct wasm call to the emitted function (win/env threaded).
                    Some(widx) => {
                        code.push(OP_LOCAL_GET);
                        uleb(code, 0); // win
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        for a in args {
                            get(code, cx, *a);
                        }
                        code.push(OP_CALL);
                        uleb(code, widx as u64);
                        // Results pushed in order; pop into destination locals in reverse.
                        for i in (0..n_results).rev() {
                            code.push(OP_LOCAL_SET);
                            uleb(code, cx.local_of[k][next_val + i] as u64);
                        }
                    }
                    // Cross-tier: `func` is an interp leaf. Marshal args as i64 slots into the env
                    // scratch, call `env.call_interp(func, args_ptr)` (the engine runs it on the
                    // bytecode interpreter and writes results back to the same slots), then reload.
                    None => {
                        if !interp_leaf[*func as usize] {
                            return Err(Error::Unsupported("call to a non-emitted, non-leaf func"));
                        }
                        if args.len().max(n_results) > XCALL_MAX_SLOTS {
                            return Err(Error::Unsupported("cross-tier call arity too large"));
                        }
                        // Store each arg to env + ENV_SCRATCH_OFF + i*8 (widen an i32 to the slot).
                        for (i, a) in args.iter().enumerate() {
                            code.push(OP_LOCAL_GET);
                            uleb(code, 1); // env
                            code.push(OP_I32_CONST);
                            sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                            code.push(0x6a); // i32.add → slot addr
                            get(code, cx, *a);
                            if callee.params[i] == ValType::I32 {
                                code.push(0xad); // i64.extend_i32_u
                            }
                            code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8
                        }
                        // env.call_interp(func_svm_idx, args_ptr = env + ENV_SCRATCH_OFF).
                        code.push(OP_I32_CONST);
                        sleb32(code, *func as i32);
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        code.push(OP_I32_CONST);
                        sleb32(code, ENV_SCRATCH_OFF as i32);
                        code.push(0x6a); // i32.add
                        code.push(OP_CALL);
                        uleb(code, 1); // func 1 = env.call_interp
                                       // Load results back from the scratch slots (narrow to i32 where needed).
                        for i in 0..n_results {
                            code.push(OP_LOCAL_GET);
                            uleb(code, 1); // env
                            code.push(OP_I32_CONST);
                            sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                            code.push(0x6a); // i32.add
                            code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8
                            if callee.results[i] == ValType::I32 {
                                code.push(0xa7); // i32.wrap_i64
                            }
                            code.push(OP_LOCAL_SET);
                            uleb(code, cx.local_of[k][next_val + i] as u64);
                        }
                    }
                }
                next_val += n_results;
            }
            // A funcref is the function index as plain `i32` data (§3c) — `RefFunc { func }` ⇒
            // `i32.const func`. The value it feeds a `CallIndirect` is masked into the table there.
            Inst::RefFunc { func } => {
                code.push(OP_I32_CONST);
                sleb32(code, *func as i32);
                set_result(cx, code, k, &mut next_val);
            }
            // Indirect call through the funcref table (§3c). Push win/env/args, then the masked table
            // index (`idx & (table_size - 1)` — exactly `dispatch_indirect`'s `idx & (len - 1)`), and
            // `call_indirect` the declared signature: wasm's built-in signature check is the type-id
            // check (a mismatch traps `IndirectCallType`); a null padding slot traps too (an empty
            // interpreter slot). No fuel debit here — the callee debits on entry to its own loop.
            Inst::CallIndirect { ty, idx, args } => {
                let n_results = ty.results.len();
                code.push(OP_LOCAL_GET);
                uleb(code, 0); // win
                code.push(OP_LOCAL_GET);
                uleb(code, 1); // env
                for a in args {
                    get(code, cx, *a);
                }
                get(code, cx, *idx);
                code.push(OP_I32_CONST);
                sleb32(code, (table_size - 1) as i32);
                code.push(0x71); // i32.and → mask into the table
                code.push(0x11); // call_indirect
                uleb(code, indirect_type_index(types, ty)? as u64);
                uleb(code, 0); // table index 0
                for i in (0..n_results).rev() {
                    code.push(OP_LOCAL_SET);
                    uleb(code, cx.local_of[k][next_val + i] as u64);
                }
                next_val += n_results;
            }
            // ---- §17 SIMD (v128) — the in-subset core lane ops (opcode helpers above) ----
            Inst::ConstV128(bytes) => {
                emit_simd(code, 12); // v128.const
                code.extend_from_slice(bytes);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::V128Load { addr, offset, .. } => {
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    16,
                    mapped,
                );
                emit_simd(code, 0); // v128.load
                code.extend_from_slice(&[0x00, 0x00]); // align=1, offset=0 (offset folded in)
                set_result(cx, code, k, &mut next_val);
            }
            Inst::V128Store {
                addr,
                value,
                offset,
                ..
            } => {
                emit_confine(
                    cx,
                    code,
                    cx.local_of[k][*addr as usize],
                    *offset,
                    16,
                    mapped,
                );
                get(code, cx, *value);
                emit_simd(code, 11); // v128.store
                code.extend_from_slice(&[0x00, 0x00]);
            }
            Inst::Splat { shape, a } => {
                get(code, cx, *a);
                emit_simd(code, vsplat_sub(*shape));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ExtractLane {
                shape,
                lane,
                signed,
                a,
            } => {
                get(code, cx, *a);
                emit_simd(code, vextract_sub(*shape, *signed));
                code.push(*lane); // lane immediate
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ReplaceLane { shape, lane, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, vreplace_sub(*shape));
                code.push(*lane);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VIntBin { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vintbin_sub(*shape, *op).ok_or(Error::Unsupported("v128 int bin shape/op"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VIntCmp { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vintcmp_sub(*shape, *op).ok_or(Error::Unsupported("v128 int cmp shape/op"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VFloatCmp { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vfloatcmp_sub(*shape, *op).ok_or(Error::Unsupported("v128 float cmp shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VShift { shape, op, a, amt } => {
                get(code, cx, *a);
                get(code, cx, *amt);
                emit_simd(
                    code,
                    vshift_sub(*shape, *op).ok_or(Error::Unsupported("v128 shift shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VIntUn { shape, op, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    vintun_sub(*shape, *op).ok_or(Error::Unsupported("v128 int un shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VSatBin { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vsatbin_sub(*shape, *op).ok_or(Error::Unsupported("v128 sat shape/op"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VAvgr { shape, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vavgr_sub(*shape).ok_or(Error::Unsupported("v128 avgr shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VPopcnt { a } => {
                get(code, cx, *a);
                emit_simd(code, 98); // i8x16.popcnt
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VConvert { op, a } => {
                get(code, cx, *a);
                emit_simd(code, vconvert_sub(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VPMinMax { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vpminmax_sub(*shape, *op).ok_or(Error::Unsupported("v128 pminmax shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VFloatBin { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vfloatbin_sub(*shape, *op).ok_or(Error::Unsupported("v128 float bin shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VFloatUn { shape, op, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    vfloatun_sub(*shape, *op).ok_or(Error::Unsupported("v128 float un shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VBitBin { op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, vbitbin_sub(*op));
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VNot { a } => {
                get(code, cx, *a);
                emit_simd(code, 77); // v128.not
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Bitselect { a, b, mask } => {
                get(code, cx, *a);
                get(code, cx, *b);
                get(code, cx, *mask);
                emit_simd(code, 82); // v128.bitselect
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VAnyTrue { a } => {
                get(code, cx, *a);
                emit_simd(code, 83); // v128.any_true
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VAllTrue { shape, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    valltrue_sub(*shape).ok_or(Error::Unsupported("v128 all_true shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VBitmask { shape, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    vbitmask_sub(*shape).ok_or(Error::Unsupported("v128 bitmask shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Shuffle { lanes, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, 13); // i8x16.shuffle
                code.extend_from_slice(lanes); // 16 lane-index immediates
                set_result(cx, code, k, &mut next_val);
            }
            Inst::Swizzle { a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, 14); // i8x16.swizzle
                set_result(cx, code, k, &mut next_val);
            }
            // ---- simd2: the widening / reduction family ----
            Inst::VWiden { shape, op, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    vwiden_sub(*shape, *op).ok_or(Error::Unsupported("v128 widen shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VNarrow { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vnarrow_sub(*shape, *op).ok_or(Error::Unsupported("v128 narrow shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VExtMul { shape, op, a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(
                    code,
                    vextmul_sub(*shape, *op).ok_or(Error::Unsupported("v128 extmul shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VExtAddPairwise { shape, signed, a } => {
                get(code, cx, *a);
                emit_simd(
                    code,
                    vextadd_sub(*shape, *signed).ok_or(Error::Unsupported("v128 extadd shape"))?,
                );
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VDot { a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, 186); // i32x4.dot_i16x8_s
                set_result(cx, code, k, &mut next_val);
            }
            Inst::VQ15MulrSat { a, b } => {
                get(code, cx, *a);
                get(code, cx, *b);
                emit_simd(code, 130); // i16x8.q15mulr_sat_s
                set_result(cx, code, k, &mut next_val);
            }
            Inst::SimdWidthBytes => {
                // Fixed-128 MVP: the constant 16 on every backend (deterministic across the oracle).
                code.push(OP_I32_CONST);
                sleb32(code, 16);
                set_result(cx, code, k, &mut next_val);
            }
            _ => return Err(Error::Unsupported("instruction outside the v1 subset")),
        }
    }
    debug_assert_eq!(next_val, value_types.len());

    match &b.term {
        Terminator::Br { target, args } => {
            emit_edge(cx, code, k, *target, args);
            cx.br_dispatch(code);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            get(code, cx, *cond);
            code.push(OP_IF);
            code.push(BLOCKTYPE_VOID);
            cx.depth += 1;
            emit_edge(cx, code, k, *then_blk, then_args);
            code.push(OP_ELSE);
            emit_edge(cx, code, k, *else_blk, else_args);
            code.push(OP_END);
            cx.depth -= 1;
            cx.br_dispatch(code);
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            // One landing block per edge (targets then default); each edge assigns its own args.
            let arms: Vec<&svm_ir::Edge> =
                targets.iter().chain(core::iter::once(default)).collect();
            for _ in &arms {
                code.push(OP_BLOCK);
                code.push(BLOCKTYPE_VOID);
                cx.depth += 1;
            }
            get(code, cx, *idx);
            code.push(OP_BR_TABLE);
            uleb(code, targets.len() as u64);
            for j in 0..targets.len() {
                uleb(code, j as u64);
            }
            uleb(code, targets.len() as u64); // default = the outermost landing block
            for (j, (target, args)) in arms.iter().enumerate() {
                code.push(OP_END);
                cx.depth -= 1;
                emit_edge(cx, code, k, *target, args);
                cx.br_dispatch(code);
                // The br above leaves this position unreachable; the next `end` (or code) follows.
                let _ = j;
            }
        }
        Terminator::Return(vals) => {
            for v in vals {
                get(code, cx, *v);
            }
            code.push(OP_RETURN);
        }
        Terminator::Unreachable => {
            code.push(OP_UNREACHABLE);
        }
        // Tail calls. A tail call's callee results equal the caller's results (the verifier guarantees
        // it), so the callee's return type matches this emitted function's — the exact condition
        // `return_call`/`return_call_indirect` validate against. Same-tier (emitted callee) and indirect
        // tail calls lower to those **native tail-call opcodes**, which reuse the caller's frame (O(1)
        // stack) — matching the interpreter's frame-reusing `Op::TailCall`, so an unbounded tail loop
        // runs in constant space on both tiers instead of overflowing the wasm stack. The **cross-tier**
        // case can't: its result comes back from the host via `env.call_interp`, so it stays an ordinary
        // call + `return` (a bounded, one-deep bounce — no frame to reuse anyway).
        Terminator::ReturnCall { func, args } => {
            let callee = &m.funcs[*func as usize];
            let n_results = callee.results.len();
            match wasm_of[*func as usize] {
                // Same-tier: a native `return_call` to the emitted function (win/env threaded).
                Some(widx) => {
                    code.push(OP_LOCAL_GET);
                    uleb(code, 0); // win
                    code.push(OP_LOCAL_GET);
                    uleb(code, 1); // env
                    for a in args {
                        get(code, cx, *a);
                    }
                    code.push(OP_RETURN_CALL);
                    uleb(code, widx as u64);
                }
                // Cross-tier: marshal args into the env scratch, `env.call_interp`, load results back
                // onto the stack, then return (the tail-call form of the mid-block cross-tier sequence).
                None => {
                    if !interp_leaf[*func as usize] {
                        return Err(Error::Unsupported(
                            "tail call to a non-emitted, non-leaf func",
                        ));
                    }
                    if args.len().max(n_results) > XCALL_MAX_SLOTS {
                        return Err(Error::Unsupported("cross-tier tail-call arity too large"));
                    }
                    for (i, a) in args.iter().enumerate() {
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        code.push(OP_I32_CONST);
                        sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                        code.push(0x6a); // i32.add → slot addr
                        get(code, cx, *a);
                        if callee.params[i] == ValType::I32 {
                            code.push(0xad); // i64.extend_i32_u
                        }
                        code.extend_from_slice(&[0x37, 0x03, 0x00]); // i64.store align=8
                    }
                    code.push(OP_I32_CONST);
                    sleb32(code, *func as i32);
                    code.push(OP_LOCAL_GET);
                    uleb(code, 1); // env
                    code.push(OP_I32_CONST);
                    sleb32(code, ENV_SCRATCH_OFF as i32);
                    code.push(0x6a); // i32.add
                    code.push(OP_CALL);
                    uleb(code, 1); // func 1 = env.call_interp
                    for i in 0..n_results {
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        code.push(OP_I32_CONST);
                        sleb32(code, (ENV_SCRATCH_OFF + i as u64 * 8) as i32);
                        code.push(0x6a); // i32.add
                        code.extend_from_slice(&[0x29, 0x03, 0x00]); // i64.load align=8
                        if callee.results[i] == ValType::I32 {
                            code.push(0xa7); // i32.wrap_i64
                        }
                    }
                    code.push(OP_RETURN);
                }
            }
        }
        // Indirect tail call: push win/env/args, mask the index into the identity table, then a native
        // `return_call_indirect` on the declared signature (wasm's signature check = the §3c type-id
        // check) — frame-reusing like the direct form. A cross-tier target resolves to its trampoline
        // slot (which itself bounces to `env.call_interp`); tail-calling the trampoline is still correct.
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            code.push(OP_LOCAL_GET);
            uleb(code, 0); // win
            code.push(OP_LOCAL_GET);
            uleb(code, 1); // env
            for a in args {
                get(code, cx, *a);
            }
            get(code, cx, *idx);
            code.push(OP_I32_CONST);
            sleb32(code, (table_size - 1) as i32);
            code.push(0x71); // i32.and → mask into the table
            code.push(OP_RETURN_CALL_INDIRECT);
            uleb(code, indirect_type_index(types, ty)? as u64);
            uleb(code, 0); // table index 0
        }
    }
    let _ = f;
    Ok(())
}

fn set_result(cx: &FnCtx, code: &mut Vec<u8>, k: usize, next_val: &mut usize) {
    code.push(OP_LOCAL_SET);
    uleb(code, cx.local_of[k][*next_val] as u64);
    *next_val += 1;
}
