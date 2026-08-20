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
//! - `env: i32` — the engine-side environment cell: cross-tier `call_interp` marshals its i64
//!   arg/result slots here (offset 16 on). Offset 0 is a legacy fuel slot the shipped emitter no
//!   longer debits (see below); it stays reserved so the scratch layout is unchanged.
//!
//! **Fuel is debited from a mutable wasm `i64` global** (exported `"fuel"`),
//! at the **IR-anchored safepoints** — one per function entry and one per taken back-edge — matching
//! the tree-walk/bytecode/Cranelift oracle exactly (INVARIANTS.md #9), so the emitted wasm traps
//! `OutOfFuel` at the *identical* safepoint for any budget (`tests/differential.rs` asserts exact
//! parity, not just the trap kind). A global (not a linear-memory cell) because no
//! guest memory store can alias it, so V8/TurboFan keeps the counter in a register across a hot
//! loop instead of reloading it every iteration — measured **1.5–2.8× on hot integer loops**, up to
//! the no-fuel bound (BROWSER.md § "Fuel in a global"). The global self-initializes to the standard
//! `1<<61` region budget, so a host that seeds the old `env`-cell no-ops harmlessly; a host wanting
//! a tighter bound (or to re-arm across regions on a reused instance) sets the `"fuel"` export
//! before the call. Fuel here is per-region/per-vCPU (as the linear-memory cell already was — the
//! threads tier allocates one budget per Worker); cross-thread preemption rides the separate
//! `epoch_cell` (svm-run), never this counter.
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
//! local. A relooper for reducible CFGs is a planned upgrade, not a correctness need — and measured
//! low-ROI so far: it made no V8 difference on Lua's giant reducible function (BROWSER.md), and the
//! per-dispatch fuel cost it also carried is now removed by the fuel global. It would only pay on
//! *multi-block-per-trip* loops, where re-dispatching each block both costs and defeats the fuel
//! global's register allocation (the `branchy` row in BROWSER.md § "Fuel in a global").
//!
//! Proven by `tests/differential.rs`: every kernel runs on the bytecode engine (the oracle) and on
//! the emitted wasm under `wasmi`, comparing results **and trap kinds**.

#![forbid(unsafe_code)]

use svm_ir::bounds::{in_window, ub_at, ub_of, UB_TOP};
use svm_ir::cap_id;
use svm_ir::{
    AtomicRmwOp, BinOp, Block, CmpOp, ConvOp, Func, FuncType, Inst, IntTy, IntUnOp, LoadOp, Module,
    StoreOp, Terminator, ValIdx, ValType, DEFAULT_RESERVED_LOG2,
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

/// The wasm global index of the fuel counter — global 0, emitted first so the `mapped` global follows
/// it at index 1 (see [`MAPPED_GLOBAL_IDX`]).
const FUEL_GLOBAL_IDX: u32 = 0;
/// The fuel global's instantiation-time default — the standard per-region budget every host seed
/// site writes (`1 << 61`). Because the global self-initializes to this, an emitted module runs
/// with the usual generous bound even if the host never touches the `"fuel"` export; a host that
/// wants a *tighter* bound (or to re-arm the counter across regions on a reused instance) sets the
/// export to a smaller value before the call. This keeps fuel-in-global a drop-in for the shipped
/// linear-memory-cell seeding: the old `env`-cell write becomes a no-op, the budget is unchanged.
const FUEL_DEFAULT: i64 = 1 << 61;

/// The wasm global index of the **live window `mapped` size** (#717 / issue: wasm-JIT confines against
/// the compile-time `mapped`). The emitted bounds check (`emit_confine`/`emit_span_check`) reads this
/// global via `global.get` instead of a baked `1 << size_log2`, so an access into memory the guest grew
/// at runtime via `vm_map` no longer spuriously faults on the JIT where the interpreter admits it. The
/// global self-initializes to the emit-time `1 << size_log2`, so a host that never grows the window sees
/// behavior **identical** to the old constant; a growing host writes the live size through the exported
/// `"mapped"` global (kept in sync by the `vm_map` cross-tier handler). It sits *after* the fuel global
/// (index 0), so its index is `1`.
const MAPPED_GLOBAL_IDX: u32 = 1;

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
/// The §14 capability ops the wasm tier lowers to a **host-driver bounce** (instead of failing
/// out-of-subset): INSTANTIATOR (iface 6) `instantiate` (op 0) and `join` (op 1) — the VM-in-VM
/// primitive (`DESIGN.md` §14; `svm_ir::cap_id::INSTANTIATOR`). The child
/// vCPU spawn/join happens host-side (as the interpreter surfaces `VcpuStop::Instantiate`), so the
/// emitted code just marshals the args to an `env.instantiate`/`env.join` import. Other §14 ops
/// (address-space, coroutines) are not lowered yet — they stay out-of-subset (fail-closed).
fn is_nested_cap(type_id: u32, op: u32) -> bool {
    type_id == cap_id::INSTANTIATOR && (op == 0 || op == 1 || op == 17)
}

/// CONSOLIDATION.md §3c.3 — does the module use the config-record spawn (`Instantiator` op 17)?
/// When it does (and only then), nested mode appends the `env.instantiate_rec` import as func
/// import 8, shifting the emitted-function base to 9 for that module alone: existing modules
/// keep their exact import set, so no driver (wasmi harness or browser JS) changes until it
/// actually loads an op-17 module.
fn module_uses_rec(m: &Module) -> bool {
    m.funcs.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::CapCall {
                        type_id: cap_id::INSTANTIATOR,
                        op: 17,
                        ..
                    }
                )
            })
        })
    })
}

/// The §14 ADDRESS_SPACE (iface 5) ops a nested unit may reach as **outlined cross-tier leaves**
/// (via the one existing `env.call_interp` transport — no new imports, per the CONSOLIDATION.md §0
/// yardstick): `page_size` (op 3, a pure query) and `sub` (op 4, attenuation-minting — it carves a
/// child window handle without changing any page state). `map`/`unmap`/`protect` (ops 0/1/2) are
/// deliberately excluded: they change page state that *subsequent emitted accesses* must honor, and
/// the wasm tier's confinement is mask-only — an emitted load would sail through an unmapped page the
/// interpreter traps. They stay out-of-subset (fail-closed to the interpreter), deferred with the
/// D40/§13 page-enforcement question.
fn is_nested_leaf_cap(type_id: u32, op: u32) -> bool {
    type_id == cap_id::ADDRESS_SPACE && (op == 3 || op == 4)
}

/// Outline each [`is_nested_leaf_cap`] `cap.call` into an appended int-signature wrapper (exactly
/// [`outline_cap_calls`]'s rewrite, restricted to that allowlist) so a §14 unit's entry that carves
/// its window (`sub`) or queries `page_size` becomes emittable: the wrapper stays a cross-tier leaf
/// [`compile_module_nested`] routes through `env.call_interp`. The host's `call_interp` callback must
/// carry the run's **powerbox** (the reactor-path contract, not the throwaway-window one), so the
/// wrapper's `cap.call` resolves against the same live `Host` the oracle uses — `sub`'s minted handle
/// then encodes identically on both tiers. INSTANTIATOR ops stay **inline** (the dedicated
/// `env.instantiate`/`env.join` bounce); all other cap ops are left inline and fail closed.
pub fn outline_nested_cap_calls(m: &mut Module) {
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
                    if !is_nested_leaf_cap(*type_id, *op) {
                        continue;
                    }
                    let g = base + wrappers.len() as u32;
                    // Wrapper signature: (handle: i32, ...sig.params) -> sig.results.
                    let mut params = Vec::with_capacity(1 + sig.params.len());
                    params.push(ValType::I32);
                    params.extend(sig.params.iter().copied());
                    let nparams = params.len() as u32;
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
                    let mut call_args = Vec::with_capacity(1 + args.len());
                    call_args.push(*handle);
                    call_args.extend(args.iter().copied());
                    *inst = Inst::Call {
                        func: g,
                        args: call_args,
                    };
                }
            }
        }
    }
    m.funcs.extend(wrappers);
}

fn block_value_types(m: &Module, b: &Block, nested_caps: bool) -> Result<Vec<ValType>, Error> {
    let mut tys: Vec<ValType> = b.params.clone();
    for inst in &b.insts {
        match inst {
            // §14 instantiator bounce (opt-in): typed by the cap-call's declared results so the entry
            // that spawns a nested VM is in-subset. Gated to the lowerable ops; any other cap-call
            // still hits the catch-all below (out-of-subset).
            Inst::CapCall { type_id, op, sig, .. } if nested_caps && is_nested_cap(*type_id, *op) => {
                for r in &sig.results {
                    tys.push(*r);
                }
            }
            // §11 slice 3: thread/futex ops in a nested unit lower to host bounces (below) —
            // typed here so a spawning unit is in-subset. Gated on `nested_caps` like the cap arm.
            Inst::ThreadSpawn { .. } if nested_caps => tys.push(ValType::I32),
            Inst::ThreadJoin { .. } if nested_caps => tys.push(ValType::I64),
            Inst::MemoryWait { .. } if nested_caps => tys.push(ValType::I32),
            Inst::MemoryNotify { .. } if nested_caps => tys.push(ValType::I32),
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
            // A standalone fence orders accesses without touching memory; no SSA result.
            Inst::AtomicFence { .. } => {}
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
            // yields a `v128`, except lane-extract (the shape's scalar) and the reductions
            // any/all_true/bitmask (`i32`). The verifier already
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
    /// engine as a cross-tier leaf: a [`marshallable_sig`] signature (each arg/result fits the
    /// scratch — `i32`/`i64`/`f32`/`f64` one slot, `v128` two, #749), **memory-free**, makes no calls (a true
    /// leaf, so no transitive window/state to share), and no concurrency / capability ops. A JITted
    /// caller reaches it via `env.call_interp`.
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
    func_in_subset_caps(m, f, atomics_ok, false)
}

/// [`func_in_subset`] with the `nested_caps` typing switch threaded through: when `true`, the §14
/// instantiator bounce and §11 thread/futex ops type as in-subset (they lower to host bounces), so a
/// nesting/threading function counts as emittable. `false` is the plain integer-compute subset.
fn func_in_subset_caps(m: &Module, f: &Func, atomics_ok: bool, nested_caps: bool) -> bool {
    // §12 atomics lower to a **single-threaded** load/(rmw)/store sequence (see the
    // `block_value_types` note): correct only when no contention is possible. `atomics_ok` is the
    // module-level guarantee of that (no reachable concurrency op ⇒ no second thread) — when it does
    // not hold, an atomic-using function stays off the JIT tier so the interpreter runs it with true
    // hardware atomicity (matching the tier-up model, which already routes concurrency to the interp).
    if !atomics_ok && func_uses_atomics(f) {
        return false;
    }
    f.blocks.iter().all(|b| {
        block_value_types(m, b, nested_caps)
            .is_ok_and(|tys| tys.iter().all(|t| valtype_byte(*t).is_ok()))
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

/// Whether `f` invokes a **shrinking or aliasing** window-remapping op — one that can make a
/// previously-accessible page trap, read different bytes, or split the read/write sets *during the
/// run*. Two ifaces do this:
/// - **ADDRESS_SPACE** (iface 5): `unmap` (1), `protect` (2). `map` (0) does **not** count (#717):
///   it only *adds* committed pages, and the emitted tier confines against the live `"mapped"`
///   global ([`MAPPED_GLOBAL_IDX`]) the driver re-syncs from the `VcpuEvent::TierUp` entry snapshot
///   (`Mem::scalar_extent`) before every emitted call — a grow the scalar can't represent (sparse,
///   non-`Rw` prot) makes the driver *decline* tier-up for that call, so the emitted tier never
///   over- or under-admits relative to the interpreter. The `map`-containing function itself is
///   still never emitted (a remapping `cap.call` is not in-subset); only its pure siblings are.
///   `page_size` (3) / `sub` (4) are a pure query / attenuation-mint and do **not** count either
///   (the emittable [`is_nested_leaf_cap`] set).
/// - **SHARED_REGION** (iface 4): `map` (0) aliases host backing into window pages (the emitted
///   tier would read the window's stale bytes, not the region's), `unmap` (1) drops it.
///
/// These desync the mask-only tier from the interpreter in ways a single monotone bound cannot
/// carry (an unmapped / RO page should trap; an aliased page reads other bytes), so their presence
/// forbids emitting — see [`module_uses_page_ops`].
fn func_uses_page_ops(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::CapCall {
                    type_id: cap_id::ADDRESS_SPACE,
                    op: 1..=2,
                    ..
                } | Inst::CapCall {
                    type_id: cap_id::SHARED_REGION,
                    op: 0..=1,
                    ..
                }
            )
        })
    })
}

/// **The window-remapping gate** (DESIGN.md §14 "wasm-JIT tier coverage"): does any function reach a
/// shrinking/aliasing remapping op ([`func_uses_page_ops`] — `unmap`/`protect` or a `SharedRegion`
/// `map`/`unmap`)? The wasm tier's confinement is a mask plus **one live bound** (the `"mapped"`
/// global) — an emitted access is admitted in `[0, live_mapped)` and cannot honor per-page state
/// beyond that shape, so once such an op is reachable an emitted load could sail through a page the
/// interpreter (which enforces `Mem`'s full page-protection + backing map) would trap on or back
/// with different bytes. That divergence is possible for **any** emitted memory access — not just
/// the op's own function — so a module that uses one must run **wholly on the interpreter** (emit
/// nothing). Checked module-wide (like [`module_atomics_ok`]) so it is sound for the un-rooted
/// tier-up path too, where an op reachable only via `thread.spawn` still forbids emitting.
///
/// **Not** gated: `ADDRESS_SPACE.map` (5,0) — a guest *grow*. Growth only adds committed pages
/// within the reservation the mask already clamps to, and the live `"mapped"` global carries it to
/// the emitted bounds check (per-call host sync from `Mem::scalar_extent`; unrepresentable states
/// decline tier-up) — see [`func_uses_page_ops`] (#717). This realizes the previously deferred
/// D40/§13 note: a grow introduces no was-accessible-now-different transition.
fn module_uses_page_ops(m: &Module) -> bool {
    m.funcs.iter().any(func_uses_page_ops)
}

/// Whether `f` invokes a §13 `SharedRegion` `map`/`unmap` (iface 4 ops 0/1) — the aliasing subset of
/// [`func_uses_page_ops`] that even the #750 paged mode cannot carry: a `Backed` page's bytes live
/// in the region backing, not the window, so an emitted access reads the wrong bytes no matter what
/// a trap check decides. Paged mode keeps these module-gated.
fn func_uses_region_ops(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::CapCall {
                    type_id: cap_id::SHARED_REGION,
                    op: 0..=1,
                    ..
                }
            )
        })
    })
}

/// Whether `f` uses a D62 bulk-memory op — excluded from the #750 paged subset (the emitted span
/// check has no per-page walk; such functions stay on the interpreter).
fn func_uses_bulk_mem(f: &Func) -> bool {
    f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::MemCopy { .. } | Inst::MemMove { .. } | Inst::MemFill { .. }
            )
        })
    })
}

/// Cap on the **estimated** emitted body size of a single function (bytes). wasm engines reject a
/// function body over a hard limit (V8: 7,654,321 bytes), which fails `WebAssembly.compile` for the
/// *whole* module. Set well under that so the conservative estimate below (which over-estimates on
/// the shipped guests) leaves margin. The one shipped case is the SQLite VDBE dispatcher, whose
/// bulk-memory body (~7.75 MB) rejoins the subset under #1004 — kept on the interpreter here.
const MAX_EST_EMITTED_FN_BYTES: usize = 6_500_000;

/// A conservative upper-bound estimate of `f`'s emitted wasm body size. Memory ops carry the fat
/// confine + NULL-guard sequence; every block is a `br_table` dispatch target whose edges shuffle
/// the target block's params ([`emit_edge`]). Calibrated to over-estimate on the shipped cards
/// (0.52–0.84 of the true body size), so `est ≤ cap` implies the real body clears the engine limit
/// with margin. Fail-safe both ways: an under-estimate merely lets an over-limit function through to
/// a graceful `WebAssembly.compile` fallback (as before #1004), an over-estimate keeps an emittable
/// function on the interpreter (a cross-tier leaf) — never an escape (§4 confinement is unaffected).
fn est_emitted_size(f: &Func) -> usize {
    let pcount = |t: u32| f.blocks.get(t as usize).map_or(0, |b| b.params.len());
    f.blocks
        .iter()
        .map(|b| {
            let insts: usize = b
                .insts
                .iter()
                .map(|i| match i {
                    Inst::Load { .. } | Inst::Store { .. } => 128,
                    Inst::MemCopy { .. } | Inst::MemMove { .. } | Inst::MemFill { .. } => 256,
                    _ => 24,
                })
                .sum();
            let edges: usize = match &b.term {
                Terminator::BrTable {
                    targets, default, ..
                } => targets
                    .iter()
                    .chain(std::iter::once(default))
                    .map(|e| 8 + 6 * pcount(e.0))
                    .sum(),
                Terminator::BrIf {
                    then_blk, else_blk, ..
                } => 8 + 6 * (pcount(*then_blk) + pcount(*else_blk)),
                Terminator::Br { target, .. } => 8 + 6 * pcount(*target),
                _ => 8,
            };
            8 + insts + edges
        })
        .sum()
}

/// Drop functions whose estimated emitted body exceeds [`MAX_EST_EMITTED_FN_BYTES`] from `in_subset`
/// — they stay on the interpreter (a cross-tier leaf under the reactor/tier-up paths, exactly as the
/// pre-#1004 bulk-mem exclusion left the SQLite dispatcher). Shared by every subset computation so
/// the size valve is uniform.
fn cap_oversized(m: &Module, in_subset: &mut [bool]) {
    for (i, f) in m.funcs.iter().enumerate() {
        if in_subset[i] && est_emitted_size(f) > MAX_EST_EMITTED_FN_BYTES {
            in_subset[i] = false;
        }
    }
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
    if !marshallable_sig(f) || f.uses_concurrency() {
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
/// Whether every param/result of `f` fits the cross-tier scratch — the ABI
/// [`emit_slot_store`]/[`emit_slot_load`] encode and every `env.call_interp` servicer decodes:
/// `i32` (widened), `i64`, `f32` (low 4 bytes), `f64` in one 8-byte slot each, and `v128` across
/// **two** consecutive slots (#749 — slot offsets are the running [`slot_off`] of the signature,
/// not `i*8`). Only `ref`/`cap` values stay unmarshallable (interpreter-tier only).
fn marshallable_sig(f: &Func) -> bool {
    f.params.iter().chain(&f.results).all(|t| {
        matches!(
            t,
            ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 | ValType::V128
        )
    })
}

/// Scratch slots one cross-tier value occupies: `v128` spans two 8-byte slots (#749), all else one.
fn slots_of(ty: ValType) -> u64 {
    match ty {
        ValType::V128 => 2,
        _ => 1,
    }
}

/// Total scratch slots a signature side occupies — the arity the [`XCALL_MAX_SLOTS`] guards bound.
fn slot_count(types: &[ValType]) -> u64 {
    types.iter().map(|t| slots_of(*t)).sum()
}

/// Byte offset of value `i` in the cross-tier scratch: the slots of everything before it. Params
/// and results each start at offset 0 (results overlay the arg slots, as ever); both the emitter
/// and every `env.call_interp` servicer compute this same running layout from the signature.
fn slot_off(types: &[ValType], i: usize) -> u64 {
    types[..i].iter().map(|t| slots_of(*t) * 8).sum()
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
    let mut in_subset: Vec<bool> = m
        .funcs
        .iter()
        .map(|f| func_in_subset(m, f, atomics_ok))
        .collect();
    // #1004: a `__null_guard`-marked module's bulk-memory span check now carries the guard low bound
    // ([`emit_span_check`]), so bulk-mem functions stay in-subset and emit — the #964 exclusion is
    // retired. (Unmarked modules were never affected: their span check emits byte-identically.)
    // The size valve then keeps a rare over-limit body (the SQLite VDBE dispatcher) off the wasm
    // tier so the whole-module emit clears the engine's per-function limit.
    cap_oversized(m, &mut in_subset);
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

/// Compile every function of a **verified** `m` into one wasm module (whole-module, all in-subset),
/// importing a **non-shared** `env.memory` — the differential/link helper. Pass `shared_memory` to
/// [`compile_module_with`] for the browser threads-build link target (shared `env.memory`); the
/// emitted code is otherwise byte-identical (only the memory-import limits differ), so the
/// `tests/differential.rs` run under `wasmi` — which has no shared memory — covers both.
pub fn compile_module(m: &Module) -> Result<Vec<u8>, Error> {
    compile_module_with(m, false)
}

/// Two imported functions precede every emitted function, so a defined function's wasm index is
/// `IMPORTED_FUNCS + its position among the emitted functions`.
const IMPORTED_FUNCS: u32 = 2;
/// In §14 `nested_caps` mode two more func imports follow `env.call_interp` — `env.instantiate` (2)
/// and `env.join` (3) — so emitted functions start at `IMPORTED_FUNCS + 2 = 4`. These indices are
/// valid only in that mode (the caller sizes `wasm_of` with the matching base).
const INSTANTIATE_IMPORT_IDX: u32 = 2;
const JOIN_IMPORT_IDX: u32 = 3;
/// §11 thread/futex host bounces (CONSOLIDATION.md §11 slice 3): func imports 4-7 in nested mode.
/// The servicer supplies the unit's module context (it knows which module it instantiated), so the
/// imports carry none. `thread_join`/`mem_wait` block the calling thread — legal on the par tiers,
/// where every vCPU is a real thread (the Worker model; `worker.js` already blocks in `Atomics.wait`).
const THREAD_SPAWN_IMPORT_IDX: u32 = 4;
const THREAD_JOIN_IMPORT_IDX: u32 = 5;
const MEM_WAIT_IMPORT_IDX: u32 = 6;
const MEM_NOTIFY_IMPORT_IDX: u32 = 7;
/// Imported-func count in `nested_caps` mode (`env.trap`, `env.call_interp`, `env.instantiate`,
/// `env.join`, `env.thread_spawn`, `env.thread_join`, `env.mem_wait`, `env.mem_notify`).
const NESTED_IMPORTED_FUNCS: u32 = 8;
/// §3c.3 — `env.instantiate_rec` (the config-record spawn bounce), emitted **conditionally** as
/// func import 8 only when [`module_uses_rec`]; the emitted-function base is then 9.
const INSTANTIATE_REC_IMPORT_IDX: u32 = 8;
/// `env.call_interp` scratch: the cross-tier call marshals its arg/result slots starting at
/// this byte offset in the `env` cell (past the `i64` fuel counter at 0). The host must allocate the
/// `env` cell at least [`ENV_CELL_BYTES`] large.
const ENV_SCRATCH_OFF: u64 = 16;
/// Max 8-byte slots the cross-tier scratch holds (a `v128` occupies two, #749; a call whose
/// params-or-results need more slots than this is refused — 64 is absurdly generous for a
/// function signature).
const XCALL_MAX_SLOTS: usize = 64;
/// Bytes the host must allocate for the `env` cell: the `i64` fuel counter + the cross-tier scratch.
pub const ENV_CELL_BYTES: usize = ENV_SCRATCH_OFF as usize + XCALL_MAX_SLOTS * 8;

/// Compile every function of a **verified** `m` into one wasm module (whole-module, all-integer).
/// Exports `f{i}` per SVM function; imports `env.memory` (shared iff `shared_memory`), `env.trap`,
/// and `env.call_interp`. Returns [`Error::Unsupported`] if *any* function is outside the v1 subset
/// — for a partly-emittable guest use the [`compile_jit`] front door.
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
    emit_module(
        m,
        shared_memory,
        &emitted,
        &wasm_of,
        &a.interp_leaf,
        None,
        false,
        None,
        svm_ir::module_null_guard(m), // #964: marked modules emit guarded on every entry
    )
}

/// Emit a **§14 VM-in-VM** unit: like [`compile_module_with`], but a `cap.call` to INSTANTIATOR
/// `instantiate` (op 0) / `join` (op 1) is lowered to a host-driver bounce (`env.instantiate` /
/// `env.join`) instead of failing out-of-subset — so a unit whose entry spawns a nested confined VM
/// runs on the wasm tier, the child spawn/join happening host-side exactly as the interpreter surfaces
/// `VcpuStop::Instantiate`. Two extra func imports precede the emitted functions, so their wasm indices
/// start at `NESTED_IMPORTED_FUNCS`. A function outside the nested subset is kept as a **cross-tier
/// leaf** via `env.call_interp` iff its signature is all-integer — the [`outline_nested_cap_calls`]
/// ADDRESS_SPACE wrappers ride this (run them with a powerbox-carrying callback) — else `Err`; the
/// entry (func 0) must itself be emittable.
/// [`compile_module_nested`] plus the per-function emit map (like [`compile_module_tierup`]'s
/// `emitted`): `eligible[i]` iff `f{i}` is an emitted export a host may call directly — the browser's
/// per-entry `svm_par_inst_eligible` gate. A cross-tier leaf (an outlined wrapper) is `false`.
pub fn compile_module_nested_with_eligibility(
    m: &Module,
    shared_memory: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    let n = m.funcs.len();
    // Track 3 (c)+(a): a window-remapping op (`map`/`unmap`/`protect`, `SharedRegion` map/unmap)
    // anywhere in the module makes *every* emitted mask-only access unsound (see
    // [`module_uses_page_ops`]). Fail closed so a direct caller (the browser's per-instance nested
    // codegen) falls back to its interpreter path — [`compile_nested`] gates this up front instead.
    if module_uses_page_ops(m) {
        return Err(Error::Unsupported(
            "window-remapping op in a nested unit (needs the interpreter tier)",
        ));
    }
    // Classify each function: in the nested subset ⇒ emitted; otherwise it must qualify as an
    // int-signature cross-tier leaf (the [`outline_nested_cap_calls`] ADDRESS_SPACE wrappers) reached
    // via `env.call_interp` — whose host callback must therefore carry the run's powerbox (the
    // reactor-path contract, not the throwaway-window one). Anything else fails closed.
    let nested_ok = |f: &Func| {
        f.blocks.iter().all(|b| {
            block_value_types(m, b, true)
                .is_ok_and(|tys| tys.iter().all(|t| valtype_byte(*t).is_ok()))
        })
    };
    // #1004: a marked granted unit emits with the NULL guard, and its bulk-mem functions now carry
    // the guard low bound in their span check ([`emit_span_check`]) — so they emit like any other
    // in-subset function instead of falling to the cross-tier leaf path.
    let null_guard = svm_ir::module_null_guard(m);
    let mut wasm_of: Vec<Option<u32>> = vec![None; n];
    let mut interp_leaf = vec![false; n];
    let mut emitted: Vec<usize> = Vec::new();
    for (i, f) in m.funcs.iter().enumerate() {
        if nested_ok(f) {
            wasm_of[i] =
                Some(NESTED_IMPORTED_FUNCS + module_uses_rec(m) as u32 + emitted.len() as u32);
            emitted.push(i);
        } else if f.uses_fibers() {
            // A fiber (`cont.*`/`suspend`) can't be emitted (no wasm frame unwind) and must not become
            // a cross-tier leaf: called synchronously from an emitted frame, a suspend across that seam
            // can't unwind. Fail closed — [`compile_nested`] routes such a unit interpreter-driven,
            // where the interpreter owns the frame and fibers run correctly.
            return Err(Error::Unsupported(
                "fiber op in a nested unit (needs the interp-driven tier)",
            ));
        } else if marshallable_sig(f)
            && f.params
                .iter()
                .chain(&f.results)
                .all(|t| !matches!(t, ValType::V128))
        {
            interp_leaf[i] = true;
        } else {
            // Not nested-emittable and carries a `v128`/`ref`/`cap` in its signature. A nested
            // unit's `env.call_interp` lands in the vCPU's `bounce_call`, whose `&[i64]` slot
            // decode is scalar-only — `v128` marshals two-slot on the module-internal cross-tier
            // ABI (#749) but NOT on this §22 transport — so it can be neither emitted nor run
            // cross-tier here. Fail closed (the unit runs whole-interpreter instead).
            return Err(Error::Unsupported(
                "v128-signature function outside the nested subset",
            ));
        }
    }
    if wasm_of.first().copied().flatten().is_none() {
        return Err(Error::Unsupported("nested entry outside the subset"));
    }
    let eligible: Vec<bool> = wasm_of.iter().map(|w| w.is_some()).collect();
    let wasm = emit_module(
        m,
        shared_memory,
        &emitted,
        &wasm_of,
        &interp_leaf,
        None,
        true,
        None,
        null_guard,
    )?;
    Ok((wasm, eligible))
}

/// The wasm-only front of [`compile_module_nested_with_eligibility`] (the original signature).
pub fn compile_module_nested(m: &Module, shared_memory: bool) -> Result<Vec<u8>, Error> {
    compile_module_nested_with_eligibility(m, shared_memory).map(|(w, _)| w)
}

/// **The §14 nested front door** — the nesting analogue of [`compile_jit`], picking the drive mode
/// from the IR so a nesting parent always yields a runnable [`Artifact`] (never `Err` for a verified
/// module). Two modes, forced by wasm's inability to unwind a frame across a fiber stack switch:
///
/// - **Nothing reachable uses a fiber** → the [`compile_module_nested`] emit: the parent's
///   `instantiate`/`join`/`thread`/`futex` ops lower to host bounces and the host calls `f{0}`
///   directly ([`DriveMode::WasmDriven`]). Threads/futex stay on this fast path — only fibers force
///   the fallback.
/// - **A fiber is reachable** → an interpreter-driven, `nested_caps`-aware tier-up
///   ([`compile_module_tierup_caps`]): the interpreter owns the top frame (running the fibers and
///   servicing `instantiate`/`join` natively), while hot in-subset compute — including any
///   instantiate/join/thread functions — still tiers up onto emitted wasm with its bounces intact
///   ([`DriveMode::InterpDriven`]).
///
/// Both modes emit the **same** nested import set (`env.instantiate`/`join`/`thread_spawn`/…), so a
/// host driving this artifact provides that one import layout regardless of the chosen mode. This
/// folds the browser's hand-rolled nested→threaded fallback into the library. Entry is func 0 (the
/// unit entry the nested emit roots at). Outline §14 ADDRESS_SPACE `cap.call`s
/// ([`outline_nested_cap_calls`]) before calling if the host's `call_interp` carries a powerbox;
/// otherwise a `sub`/`page_size` entry simply falls to the interpreter-driven mode.
///
/// A unit that manages its own pages (`map`/`unmap`/`protect`) emits **nothing** and runs wholly on
/// the interpreter — the mask-only tier can't honor page state (Track 3 (c)+(a), see
/// [`module_uses_page_ops`]). `page_size`/`sub` (queries/attenuation) are unaffected.
pub fn compile_nested(m: &Module, shared_memory: bool) -> Result<Artifact, Error> {
    if module_uses_page_ops(m) {
        return compile_interp_only(m, shared_memory, true);
    }
    if !reachable_fibers(m, 0) {
        if let Ok((wasm, emitted)) = compile_module_nested_with_eligibility(m, shared_memory) {
            return Ok(Artifact {
                wasm,
                emitted,
                drive: DriveMode::WasmDriven { entry: 0 },
            });
        }
    }
    let (wasm, emitted) = compile_module_tierup_caps(m, shared_memory, true)?;
    Ok(Artifact {
        wasm,
        emitted,
        drive: DriveMode::InterpDriven,
    })
}

/// Whole-module emit like [`compile_module_with`], but wired for **§22 Model B2**: instead of a
/// private table the module fills itself, it *imports* the domain's one shared funcref table
/// (`env.__indirect_function_table`, sized `1 << table_log2` = `Host::jit_table_log2`) and populates
/// nothing — the host writes every slot (own funcs + `install`ed units) via `table.set`, exactly as
/// the interpreter's `DomainTable` is host-populated. This is what makes an installed unit a funcref
/// another instance's `call_indirect` can reach (old→new) at native speed. The confinement mask stays
/// a compile-time constant `1<<table_log2` (invariant I2). `table_log2` must match the reservation the
/// domain was granted with (`grant_jit_with_table` / `DomainTable::new(_, table_log2)`).
pub fn compile_module_b2(
    m: &Module,
    shared_memory: bool,
    table_log2: u32,
) -> Result<Vec<u8>, Error> {
    let a = analyze(m);
    if !a.in_subset.iter().all(|&s| s) {
        return Err(Error::Unsupported(
            "a function is outside the integer subset",
        ));
    }
    let n = m.funcs.len();
    let emitted: Vec<usize> = (0..n).collect();
    let wasm_of: Vec<Option<u32>> = (0..n).map(|i| Some(IMPORTED_FUNCS + i as u32)).collect();
    emit_module(
        m,
        shared_memory,
        &emitted,
        &wasm_of,
        &a.interp_leaf,
        Some(table_log2),
        false,
        None,
        svm_ir::module_null_guard(m), // #964: marked modules emit guarded on every entry
    )
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
/// renumbering — the same property the linker-only [`svm_ir::resolve_imports_with`] relies on).
///
/// Named manifest imports are **not** rewritten here (or anywhere — the runtime binds each import
/// to a slot at instantiation; see `CallImport` below). The transformed module must be the one
/// **both** tiers use: the emitter reads it, and the host's `call_interp` runs the wrapper on the
/// interpreter — the wrapper only exists in the outlined module.
///
/// It outlines the host-boundary ops the same way: [`Inst::CapCall`], [`Inst::CallImport`] (an
/// executable manifest import, IMPORTS.md phase 3 — the wrapper carries the `call.import` to the
/// import-capable interpreter tier, so an import-bearing guest emits without resolution or rewrite),
/// and [`Inst::CallSym`]. The §7 runtime by-name capability lookup (`cap.self.resolve`) rides through
/// the `CapCall` arm now (it is a `cap.call CAP_SELF op 2`): that matters for a `_start` that resolves
/// names at startup — otherwise pure compute + stores, so hoisting its handful of `cap.self.resolve`s
/// into cross-tier wrappers makes func 0 itself emittable — the last thing keeping a QuickJS-scale
/// guest (whose hot interpreter loop is all in-subset) off the wasm tier.
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
                    op,
                    sig,
                    args,
                } = inst
                {
                    let g = base + wrappers.len() as u32;
                    // Wrapper shape (v8): (...sig.params) -> sig.results — no handle operand;
                    // the import index is an immediate, so it stays baked into the wrapper body.
                    let params = sig.params.clone();
                    let nparams = params.len() as u32;
                    let wrapper_args: Vec<u32> = (0..nparams).collect();
                    let ret: Vec<u32> = (nparams..nparams + sig.results.len() as u32).collect();
                    let block = Block {
                        params: params.clone(),
                        insts: vec![Inst::CallImport {
                            import: *import,
                            op: *op,
                            sig: sig.clone(),
                            args: wrapper_args,
                        }],
                        term: Terminator::Return(ret),
                    };
                    wrappers.push(Func {
                        params,
                        results: sig.results.clone(),
                        blocks: vec![block],
                    });
                    let call_args = args.clone();
                    *inst = Inst::Call {
                        func: g,
                        args: call_args,
                    };
                } else if let Inst::CallSym {
                    import,
                    sig,
                    handle,
                    args,
                } = inst
                {
                    let g = base + wrappers.len() as u32;
                    // §7/§22 symbolic call: same shape as the old handle-carrying form —
                    // (handle: i32, ...sig.params) -> sig.results; the dispatch ignores the
                    // handle, but it is a live call-site register to thread through.
                    let mut params = Vec::with_capacity(1 + sig.params.len());
                    params.push(ValType::I32);
                    params.extend(sig.params.iter().copied());
                    let nparams = params.len() as u32;
                    let wrapper_args: Vec<u32> = (1..nparams).collect();
                    let ret: Vec<u32> = (nparams..nparams + sig.results.len() as u32).collect();
                    let block = Block {
                        params: params.clone(),
                        insts: vec![Inst::CallSym {
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
                }
            }
        }
    }
    m.funcs.extend(wrappers);
}

/// V8 rejects any single wasm function body larger than `kV8MaxWasmFunctionSize` (7,654,321 bytes) at
/// compile time, and one over-large body makes the **whole** module unloadable. A function whose
/// emitted body exceeds this cap is therefore pulled out of the emitted set and run as a cross-tier
/// interpreter leaf instead — it executes on the bytecode engine over the shared window (reached via
/// `env.call_interp`) while the rest of the module still JITs. The threshold sits below the hard limit
/// with margin: the measured length is the function payload alone, and re-routing a caller's direct
/// call to a now-excluded callee grows the *caller* by a few bytes, so we leave headroom rather than
/// emit right up to the wall. Discovered via nifler, whose linked-in (but never-run) Nim VM
/// `rawExecute` lowers to an ~8.5 MB body (#1011).
const MAX_EMITTED_FUNC_BYTES: usize = 7_000_000;

/// Decode a ULEB128 at `p[i..]`, returning `(value, next_index)`. The bytes come from our own
/// [`emit_module`] output, so the encoding is assumed well-formed (no overlong / truncation check).
fn read_uleb(p: &[u8], mut i: usize) -> (u64, usize) {
    let (mut val, mut shift) = (0u64, 0u32);
    loop {
        let b = p[i];
        i += 1;
        val |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return (val, i);
        }
        shift += 7;
    }
}

/// Byte length of each function body in the code section (id 10) of an assembled wasm module, in emit
/// order — the [`emit_module`] order (`emitted` first, then trampolines / trap stub). Used to find a
/// body that exceeds [`MAX_EMITTED_FUNC_BYTES`]. Input is our own well-formed output, so the section
/// walk is unchecked. Returns empty if there is no code section (a module with no emitted functions).
fn code_section_body_sizes(wasm: &[u8]) -> Vec<usize> {
    let mut i = 8; // skip the 8-byte "\0asm" + version preamble
    while i < wasm.len() {
        let id = wasm[i];
        i += 1;
        let (size, after_len) = read_uleb(wasm, i);
        i = after_len;
        if id == 10 {
            let (count, mut j) = read_uleb(wasm, i);
            let mut sizes = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let (blen, after) = read_uleb(wasm, j);
                sizes.push(blen as usize);
                j = after + blen as usize;
            }
            return sizes;
        }
        i += size as usize;
    }
    Vec::new()
}

/// Compile a **whole-module reactor** guest with **widened cross-tier calls** (Doom-perf): emit every
/// reachable in-subset function to wasm and route a **direct** `Call` to any reachable, non-emitted,
/// **integer-signature** function through `env.call_interp` — not just the strict memory-free/call-free
/// `interp_leaf`s the throwaway-window path allows. A cross-tier callee here may touch memory,
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
/// a non-[`marshallable_sig`] signature (a `ref`/`cap` param/result can't be marshalled cross-tier —
/// `v128` marshals two-slot since #749), or an address-taken indirect target is itself
/// non-marshallable (can't be trampolined).
///
/// A function whose emitted body would exceed the V8 per-function byte cap ([`MAX_EMITTED_FUNC_BYTES`])
/// is pulled out of the emitted set and run as a cross-tier leaf too (#1011) — so a guest that links a
/// giant-but-cold function (nifler's ~8.5 MB Nim VM `rawExecute`) still JITs the rest of the module
/// instead of shipping a body V8 refuses to load. We can't know a body's size without emitting it, so
/// this emits, measures, excludes any over-large marshallable body, and re-emits until every emitted
/// body fits; the common case finds none and emits once. An over-large body that *can't* be crossed
/// (non-marshallable) forces [`Error::Unsupported`], dropping the guest to the fully-interpreted
/// artifact rather than an unloadable module.
pub fn compile_module_reactor(
    m: &Module,
    entry: u32,
    shared_memory: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_reactor_capped(m, entry, shared_memory, MAX_EMITTED_FUNC_BYTES)
}

/// [`compile_module_reactor`] with an explicit emitted-body byte cap. The public wrapper passes V8's
/// [`MAX_EMITTED_FUNC_BYTES`]; the parameter exists so the exclusion loop can be differential-tested
/// with a small cap (and a small module) instead of a genuine multi-megabyte function. `usize::MAX`
/// disables the size exclusion (an engine with no per-function cap).
#[doc(hidden)]
pub fn compile_module_reactor_capped(
    m: &Module,
    entry: u32,
    shared_memory: bool,
    cap: usize,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    let n = m.funcs.len();
    let a = analyze_from(m, entry);
    // `oversized[i]` — an in-subset function pulled from the emitted set because its body exceeds the
    // V8 cap (#1011). Monotone: it only grows across re-emits, so the loop runs at most once per
    // over-large function and terminates. Empty on the first pass ⇒ byte-identical emit for every guest
    // that has no over-large body.
    let mut oversized = vec![false; n];
    loop {
        // Cross-tier: reachable, not emitted (not in-subset, or pulled for size), and marshallable
        // (fits the scratch slots). Runs on the interpreter over the shared window — so, unlike
        // `interp_leaf`, memory/calls/caps are fine.
        let cross: Vec<bool> = (0..n)
            .map(|i| {
                a.reachable[i] && (!a.in_subset[i] || oversized[i]) && marshallable_sig(&m.funcs[i])
            })
            .collect();
        let ok = (entry as usize) < n
            && a.in_subset[entry as usize]
            && !oversized[entry as usize] // a wasm-driven entry must itself be emitted
            // Every reachable function must be emittable or cross-tier-callable. When the guest makes an
            // indirect call, `analyze` marks **all** functions reachable, so this also guarantees every
            // possible indirect target (including a data-segment function pointer, which no `RefFunc` scan
            // sees) is either emitted or `cross` — hence gets an identity-table slot: the emitted wasm
            // function, or a trampoline that bounces to the interpreter (see `emit_module`).
            && (0..n).all(|i| !a.reachable[i] || (a.in_subset[i] && !oversized[i]) || cross[i]);
        if !ok {
            return Err(Error::Unsupported("guest not cross-tier reactor runnable"));
        }
        let mut wasm_of: Vec<Option<u32>> = vec![None; n];
        let mut emitted: Vec<usize> = Vec::new();
        let mut emitted_bitmap = vec![false; n];
        for i in 0..n {
            if a.reachable[i] && a.in_subset[i] && !oversized[i] {
                wasm_of[i] = Some(IMPORTED_FUNCS + emitted.len() as u32);
                emitted.push(i);
                emitted_bitmap[i] = true;
            }
        }
        let wasm = emit_module(
            m,
            shared_memory,
            &emitted,
            &wasm_of,
            &cross,
            None,
            false,
            None,
            svm_ir::module_null_guard(m), // #964: marked modules emit guarded on every entry
        )?;
        // Measure the emitted bodies and pull any over-large *marshallable* one out for a re-emit as a
        // cross-tier leaf. An over-large body that can't be crossed (non-marshallable sig) has nowhere
        // to go — fail closed so `compile_jit` drops to the interpreter rather than ship a module V8
        // rejects. `code_section_body_sizes` returns bodies in emit order, so `sizes[bi]` is `emitted[bi]`.
        let sizes = code_section_body_sizes(&wasm);
        let mut progressed = false;
        for (bi, &fi) in emitted.iter().enumerate() {
            if sizes[bi] > cap {
                if marshallable_sig(&m.funcs[fi]) {
                    oversized[fi] = true;
                    progressed = true;
                } else {
                    return Err(Error::Unsupported(
                        "emitted function exceeds the V8 body cap",
                    ));
                }
            }
        }
        if !progressed {
            return Ok((wasm, emitted_bitmap));
        }
    }
}

/// Like [`compile_module_reactor`], but emit only the functions in `keep` (plus any whose signature
/// can't be marshalled cross-tier, and the entry) — the rest of the reachable in-subset set falls back
/// to the interpreter via `env.call_interp`, exactly like a non-in-subset function. This is the
/// **subset / profile-guided emit** the tier-up break-even measurement drives: `keep` is the set a run
/// actually touches (the profiler showed ~12% of functions), so the emitted module carries only the hot
/// code. A dropped function that is later called still runs correctly (on the interp), just not on wasm.
/// `keep[i]` out of range defaults to *kept* (so an under-sized `keep` degrades to whole-module emit).
pub fn compile_module_reactor_keep(
    m: &Module,
    entry: u32,
    keep: &[bool],
    shared_memory: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    let n = m.funcs.len();
    let a = analyze_from(m, entry);
    // Emit a reachable in-subset function iff it is the entry, kept, or has a signature that can't be
    // marshalled to the interpreter (so it *must* stay on wasm). Everything else reachable becomes
    // cross-tier (interp over the shared window) — the same fallback path non-in-subset functions use.
    let emitted_pred: Vec<bool> = (0..n)
        .map(|i| {
            a.reachable[i]
                && a.in_subset[i]
                && (i as u32 == entry
                    || keep.get(i).copied().unwrap_or(true)
                    || !marshallable_sig(&m.funcs[i]))
        })
        .collect();
    let cross: Vec<bool> = (0..n)
        .map(|i| a.reachable[i] && !emitted_pred[i] && marshallable_sig(&m.funcs[i]))
        .collect();
    let ok = (entry as usize) < n
        && emitted_pred[entry as usize]
        && (0..n).all(|i| !a.reachable[i] || emitted_pred[i] || cross[i]);
    if !ok {
        return Err(Error::Unsupported("keep-set not cross-tier runnable"));
    }
    let mut wasm_of: Vec<Option<u32>> = vec![None; n];
    let mut emitted: Vec<usize> = Vec::new();
    let mut emitted_bitmap = vec![false; n];
    for (i, &e) in emitted_pred.iter().enumerate() {
        if e {
            wasm_of[i] = Some(IMPORTED_FUNCS + emitted.len() as u32);
            emitted.push(i);
            emitted_bitmap[i] = true;
        }
    }
    let wasm = emit_module(
        m,
        shared_memory,
        &emitted,
        &wasm_of,
        &cross,
        None,
        false,
        None,
        svm_ir::module_null_guard(m), // #964: marked modules emit guarded on every entry
    )?;
    Ok((wasm, emitted_bitmap))
}

/// Compile a **tier-up** module for the browser threads tier (`BROWSER.md` § "wasm-JIT tier",
/// per-Worker JIT). Unlike the rooted, wasm-driven [`compile_module_reactor`], eligibility is **not** rooted at one
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
    compile_module_tierup_caps(m, shared_memory, false)
}

/// [`compile_module_tierup`] with the `nested_caps` switch: when `true`, the §14 instantiator bounce
/// and §11 thread/futex ops count as in-subset (they lower to host bounces), and the emitted functions
/// follow the nested import layout (`NESTED_IMPORTED_FUNCS`). This is the interpreter-driven fallback
/// [`compile_nested`] uses for a fiber-bearing nested unit: the interpreter owns the frame (fibers and
/// `instantiate`/`join` run there), and hot in-subset compute — including instantiate/join/thread
/// functions — still tiers up onto the emitted region with its bounces intact.
pub fn compile_module_tierup_caps(
    m: &Module,
    shared_memory: bool,
    nested_caps: bool,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_tierup_inner(m, shared_memory, nested_caps, None, None, None)
}

/// #880 — [`compile_module_tierup`] over the **shared reserved table** (§22 Model B2): the emitted
/// module *imports* `env.__indirect_function_table` (sized `1 << table_log2`, the domain's
/// reservation) instead of declaring a local identity table, and — the point — the
/// `all_in_subset || !uses_indirect` candidate restriction is **dropped**: a `call_indirect`-bearing
/// function tiers up, because under a host-populated table every slot resolves *correctly*
/// regardless of the emit split — an emitted target is a native funcref (an installed unit's `f0`,
/// another `f{i}`), an interpreter-resident target is a live-state bounce shim
/// ([`emit_slot_trampoline`]), and an empty slot is a null-funcref trap, exactly the interpreter's
/// `TABLE_EMPTY`. This is what lets a program's hot **dispatch loop** (the language-runtime shape)
/// run on the emitted tier, including the old→new edge into installed units. The host owns slot
/// population and must keep the table exact at every entry to emitted code (the single-shot pump
/// syncs at event boundaries — installs only happen between events).
///
/// #888 — this mode also **widens the cross-tier set** to the reactor's `cross` (every
/// [`marshallable_sig`] non-in-subset function, not just the strict memory-free [`interp_leaf`]),
/// collapsing the fixpoint cascade so an in-subset function that calls a memory/cap helper stays
/// emitted. **Contract:** the host must therefore service `env.call_interp` over the run's **LIVE**
/// window/powerbox/fuel (the pump's `svm_onramp_tierup_call_interp` → the live-state bounce) — the
/// same contract the bounce shims already impose — not the throwaway-window bounce the leaf-only
/// drivers use.
pub fn compile_module_tierup_b2(
    m: &Module,
    shared_memory: bool,
    table_log2: u32,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_tierup_inner(m, shared_memory, false, None, Some(table_log2), None)
}

/// #1009 — [`compile_module_tierup_b2`] **paged** (#750): the shared-reserved-table dispatch mode
/// (B2, `table_log2`), but every emitted access also consults the host-maintained page-state table
/// (`page_log2`), so a guest that write-protects its rodata (`readonly` data segments — every real
/// on-ramp card does) keeps its pure leaves eligible instead of declining the whole tier-up at the
/// per-call `scalar_extent` sync. Composes the two orthogonally: the B2 cross-tier widening and
/// indirect-dispatch lift (see [`compile_module_tierup_b2`]) with the paged per-access page check and
/// its bulk-memory exclusion (see [`compile_module_tierup_paged`]). The driver contract is the union
/// of both — service `env.call_interp` over the **live** window (B2), and before each emitted call
/// refresh the page-state table from the live map and write its base to `"pagestate"` + its coverage
/// to `"mapped"` (#750). This is what the single-shot pump uses for a rodata-bearing card.
pub fn compile_module_tierup_b2_paged(
    m: &Module,
    shared_memory: bool,
    table_log2: u32,
    page_log2: u8,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_tierup_inner(
        m,
        shared_memory,
        false,
        Some(page_log2),
        Some(table_log2),
        None,
    )
}

/// The **opt-in gated software page-check** entry (#750): like [`compile_module_tierup`], but the
/// module is compiled in **paged** mode — the shrinking page ops (`unmap`/`protect`, iface 5 ops
/// 1/2) no longer force emit-nothing, because every emitted access also consults a host-maintained
/// byte-per-page state table (`0 = Unmapped`, `1 = Rw`, `2 = Ro`; base written to the exported
/// `"pagestate"` i32 global) and traps exactly where the interpreter's `check_prot` would. The
/// driver contract extends #717's: before each emitted call, refresh the table from the live page
/// map (`Mem::map_info`) **and** set `"mapped"` to the table's byte coverage — the bound check
/// then traps everything above the table exactly where the interpreter (no entries above) faults,
/// and the page states refine within it. `page_log2` is the run's software page size, baked into
/// the emitted shift.
///
/// Deliberate paged-mode limits (fail-closed): `SharedRegion` `map`/`unmap` (iface 4) still gate
/// the whole module — a `Backed` page's bytes live outside the window, which no trap check can
/// honor — and bulk-memory functions (`mem.copy`/`move`/`fill`) stay on the interpreter (their span
/// check has no per-page walk). Every non-paged entry emits byte-identical code to before #750.
pub fn compile_module_tierup_paged(
    m: &Module,
    shared_memory: bool,
    page_log2: u8,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_tierup_inner(m, shared_memory, false, Some(page_log2), None, None)
}

/// **Experimental NULL-page guard** entry (measurement mode for the trap-on-NULL design; see the
/// tracking issue): like [`compile_module_tierup`], but every confined access additionally traps
/// when its first byte lands below `guard` — the cheap dedicated lowering (one compare +
/// never-taken branch, no table load) the design weighs against full paged mode for trapping NULL
/// dereferences. Trap-parity holds against an interpreter whose page map seeds `[0, guard)`
/// `Unmapped` (the `nullguard.rs` differential). A *bottom* guard needs no last-byte check (an
/// access starting at or above `guard` cannot reach down), and — like the page check — it is never
/// elided (an in-window proof bounds the access from above, not below). Bulk-memory functions stay
/// interpreted (the span check has no low bound), same limit as paged mode.
///
/// Since the #964 ABI landed this is the **measurement/override** entry: production compiles derive
/// the guard from the module's `__null_guard` marker on every standard entry (see
/// [`compile_module_tierup_inner`]), so a marked module needs no special entry. This one forces an
/// arbitrary `guard` on an *unmarked* module — what `paged_bench_emit` / `bench_paged.mjs` emit +
/// time as a third variant.
pub fn compile_module_tierup_nullguard(
    m: &Module,
    shared_memory: bool,
    guard: u64,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    compile_module_tierup_inner(m, shared_memory, false, None, None, Some(guard))
}

fn compile_module_tierup_inner(
    m: &Module,
    shared_memory: bool,
    nested_caps: bool,
    paged: Option<u8>,
    reserved_table_log2: Option<u32>,
    null_guard: Option<u64>,
) -> Result<(Vec<u8>, Vec<bool>), Error> {
    // #964: a `__null_guard`-marked module opts every tier into the NULL guard — derive it from the
    // marker whenever the caller didn't force one (the measurement entry still can), so the plain
    // tier-up entries stay trap-parity with the interpreter oracle, which seeds `[0, guard)`
    // `Unmapped` for marked modules. Unmarked modules keep `None` (byte-identical emit).
    let null_guard = null_guard.or_else(|| svm_ir::module_null_guard(m));
    let n = m.funcs.len();
    // Track 3 (c)+(a): a page-op module (`map`/`unmap`/`protect`) can't be accelerated on the
    // mask-only tier — an emitted access ignores per-page state the interpreter would trap on
    // (`NESTED_JIT.md`; see [`module_uses_page_ops`]). `compile_jit`/`compile_nested` gate this before
    // reaching here, but this is a **public entry** a host can call directly (the JACL browser tier-up
    // spike did — `jacl_impl/docs/SVM_BROWSER_TIERUP_FINDINGS.md`, where an emitted leaf over a
    // page-managed window trapped `MemoryFault` mid-body). Self-protect: emit nothing, exactly like
    // `compile_interp_only`, so the gate holds regardless of caller.
    //
    // Paged mode (#750) narrows the gate to `SharedRegion` aliasing only: `unmap`/`protect` are
    // carried by the emitted per-access page check instead (see `compile_module_tierup_paged`).
    let gated = match paged {
        None => module_uses_page_ops(m),
        Some(_) => m.funcs.iter().any(func_uses_region_ops),
    };
    if gated {
        let wasm_of = vec![None; n];
        let leaf = vec![false; n];
        let wasm = emit_module(
            m,
            shared_memory,
            &[],
            &wasm_of,
            &leaf,
            None,
            nested_caps,
            paged,
            null_guard,
        )?;
        return Ok((wasm, vec![false; n]));
    }
    let atomics_ok = module_atomics_ok(m);
    let mut in_subset: Vec<bool> = m
        .funcs
        .iter()
        .map(|f| func_in_subset_caps(m, f, atomics_ok, nested_caps))
        .collect();
    if paged.is_some() {
        // Paged-mode limit (#750): bulk-memory ops have no per-page walk in the emitted span check,
        // so their functions stay on the interpreter (which honors the full page map). Fail-closed:
        // dropping them from the subset can only route more code to the oracle. (The NULL guard no
        // longer shares this limit — #1004 gave the span check its guard low bound; a marked module
        // that is *also* paged still excludes them here, the per-page walk being the open problem.)
        for (i, f) in m.funcs.iter().enumerate() {
            if func_uses_bulk_mem(f) {
                in_subset[i] = false;
            }
        }
    }
    // #1004 size valve (see `cap_oversized`): a body over the engine's per-function limit stays on
    // the interpreter — a cross-tier leaf here, since the tier-up `leaf` set below admits any
    // `marshallable_sig` non-subset function.
    cap_oversized(m, &mut in_subset);
    // The cross-tier set — functions an emitted `Call`/`call_indirect` routes to `env.call_interp`.
    // Two widths, by who services the bounce:
    //   * **local table** (`reserved_table_log2 == None`): the strict [`interp_leaf`] set —
    //     memory-free, call-free, cap-free — because the leaf-only drivers run a bounce over a
    //     *throwaway* window (a memory-touching leaf would diverge from the shared one).
    //   * **shared reserved table** (#888, B2 mode — `compile_module_tierup_b2`): the reactor's
    //     `cross` set — **any** [`marshallable_sig`] non-in-subset function. This mode's host
    //     services `env.call_interp` over the run's **LIVE** window/powerbox/fuel (the pump's
    //     `svm_onramp_tierup_call_interp` → the #846/#880 live-state bounce), exactly
    //     [`compile_module_reactor`]'s contract, so a cross-tier callee may touch memory, make
    //     calls, and cap-call. Widening here collapses the fixpoint cascade below: an in-subset
    //     function that calls a memory/cap helper stays emitted instead of being dropped
    //     (#887 measured this as ~30% → ~90% static coverage on the C-family cards).
    let leaf: Vec<bool> = (0..n)
        .map(|i| {
            !in_subset[i]
                && if reserved_table_log2.is_some() {
                    marshallable_sig(&m.funcs[i])
                } else {
                    interp_leaf(&m.funcs[i])
                }
        })
        .collect();
    let all_in_subset = in_subset.iter().all(|&s| s);
    // Emitted functions follow the import block; the nested layout adds the §14/§11 bounce imports.
    let base = if nested_caps {
        NESTED_IMPORTED_FUNCS + module_uses_rec(m) as u32
    } else {
        IMPORTED_FUNCS
    };

    // Optimistic start: every in-subset function is a candidate. A `call_indirect` can dispatch to
    // any identity-table slot, so under the **local** table a function that uses one is only safe to
    // emit when every function is in-subset (all slots resolve); under the **shared reserved** table
    // (#880 — `compile_module_tierup_b2`) the restriction lifts: the host populates every slot
    // correctly (native funcref / bounce shim / trapping null), so indirect dispatch is always
    // routable and dispatch-loop functions tier up.
    let mut emit: Vec<bool> = (0..n)
        .map(|i| {
            in_subset[i]
                && (reserved_table_log2.is_some()
                    || all_in_subset
                    || !func_uses_indirect(&m.funcs[i]))
        })
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
            wasm_of[i] = Some(base + emitted.len() as u32);
            emitted.push(i);
        }
    }
    let wasm = emit_module(
        m,
        shared_memory,
        &emitted,
        &wasm_of,
        &leaf,
        reserved_table_log2,
        nested_caps,
        paged,
        null_guard,
    )?;
    Ok((wasm, emit))
}

// ---- the unified front door -----------------------------------------------------------------------

/// The embedder's **invocation intent** — the one thing that is *not* derivable from the IR (two
/// guests can have byte-identical IR and differ only in how the host chooses to drive them). Everything
/// else — which functions emit, who owns the top-level frame, where the bytecode fallback kicks in — is
/// derived from the module by [`compile_jit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Run `entry` once to completion (a batch `_start`, or the cross-engine bench's kernel).
    Batch { entry: u32 },
    /// A long-lived reactor re-entered at `entry` (`tick`) each activation.
    Reactor { entry: u32 },
    /// Threaded / no single root — vCPUs enter through `thread.spawn`, so there is no top-level
    /// wasm frame the host can own; the interpreter must drive.
    Threaded,
}

/// How the host must drive the wasm [`Artifact::wasm`] — the strategy [`compile_jit`] picked from the
/// IR. The two variants are the irreducible fork (who owns the top-level stack frame), forced by
/// wasm's inability to unwind a frame across a suspension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveMode {
    /// **Wasm owns the top-level frame:** the host calls `f{entry}` directly; a reachable function
    /// with `emitted[i] == false` is a cross-tier callee run on the bytecode interpreter over the
    /// shared window via `env.call_interp`. Chosen when the guest is rooted and nothing reachable can
    /// suspend (a wasm frame can't unwind for a stack switch).
    WasmDriven { entry: u32 },
    /// **The bytecode interpreter owns the top-level frame** and drives scheduling / suspension /
    /// threads; a direct `Call` to an `emitted[i]` function tiers up onto the emitted region. The
    /// universal fallback — it runs any guest, JIT-accelerating whatever compute it can (possibly
    /// nothing, i.e. a pure-bytecode run).
    InterpDriven,
}

/// A compiled guest plus how to drive it: the emitted wasm, the per-function **emitted** bitmap
/// (`emitted[i]` ⇒ `f{i}` is exported and runs as wasm; the rest are interpreter-serviced), and the
/// [`DriveMode`] the host must use.
pub struct Artifact {
    pub wasm: Vec<u8>,
    pub emitted: Vec<bool>,
    pub drive: DriveMode,
}

/// Whether any function reachable from `entry` uses a §12 concurrency op (`cont.*`/`suspend`/
/// `thread.*`/futex). Such a guest cannot be wasm-driven: the interpreter must own the stack so it can
/// unwind across a suspension / block a vCPU (a wasm frame can neither).
fn reachable_concurrency(m: &Module, entry: u32) -> bool {
    let a = analyze_from(m, entry);
    (0..m.funcs.len()).any(|i| a.reachable[i] && m.funcs[i].uses_concurrency())
}

/// Whether any function reachable from `entry` uses a **fiber** op (`cont.*`/`suspend`) — the one §12
/// family the nested emitter cannot lower (a wasm frame can't unwind for a stack switch), unlike
/// thread/futex ops, which the nested tier *does* emit as host bounces. This is the [`compile_nested`]
/// two-mode gate: fibers force the interpreter to own the frame (see [`Func::uses_fibers`]).
fn reachable_fibers(m: &Module, entry: u32) -> bool {
    let a = analyze_from(m, entry);
    (0..m.funcs.len()).any(|i| a.reachable[i] && m.funcs[i].uses_fibers())
}

/// Emit **nothing** — a valid wasm module of imports only, with an all-`false` `emitted` bitmap and
/// [`DriveMode::InterpDriven`]. The whole guest runs on the bytecode interpreter (the oracle), which
/// alone enforces per-page protection. This is the Track 3 (c)+(a) landing for a page-op module (see
/// [`module_uses_page_ops`]): correct by construction, since nothing the mask-only tier could emit
/// ever runs. `nested_caps` only selects the import layout (unused here, but kept uniform with the
/// caller's other artifacts).
fn compile_interp_only(
    m: &Module,
    shared_memory: bool,
    nested_caps: bool,
) -> Result<Artifact, Error> {
    let n = m.funcs.len();
    let wasm_of = vec![None; n];
    let leaf = vec![false; n];
    let wasm = emit_module(
        m,
        shared_memory,
        &[],
        &wasm_of,
        &leaf,
        None,
        nested_caps,
        None,
        svm_ir::module_null_guard(m), // #964 (vacuous here — no bodies — but kept uniform)
    )?;
    Ok(Artifact {
        wasm,
        emitted: vec![false; n],
        drive: DriveMode::InterpDriven,
    })
}

/// **The single wasm-JIT front door.** The embedder supplies only the invocation [`Shape`]; the
/// execution *strategy* is derived from the IR: a rooted, suspension-free guest is **wasm-driven**
/// (the whole hot path is emitted wasm, fastest), everything else is **interpreter-driven** with
/// per-function tier-up (the interpreter owns scheduling/suspension and lifts hot compute). Either
/// way non-emitted functions fall back to the bytecode interpreter, so this never fails to produce a
/// runnable artifact for a verified module.
///
/// This removes the strategy choice from consumers: picking wrong could previously only cost
/// performance (never correctness), so deriving it here is pure upside. The one honest parameter left
/// is the `shape` — the host's own invocation intent, which the bytes can't express.
pub fn compile_jit(m: &Module, shape: Shape, shared_memory: bool) -> Result<Artifact, Error> {
    let interp_driven = |m: &Module| -> Result<Artifact, Error> {
        let (wasm, emitted) = compile_module_tierup(m, shared_memory)?;
        Ok(Artifact {
            wasm,
            emitted,
            drive: DriveMode::InterpDriven,
        })
    };
    // A module that manages its own pages (`map`/`unmap`/`protect`) can't be accelerated on the
    // mask-only tier — an emitted access ignores page state the interpreter would trap on — so emit
    // nothing and run it wholly on the interpreter (DESIGN.md §14 "wasm-JIT tier coverage"). Checked
    // before the shape split so it holds for `Threaded`/tier-up too, not just the rooted paths.
    if module_uses_page_ops(m) {
        return compile_interp_only(m, shared_memory, false);
    }
    match shape {
        // No single top-level frame the host can own → the interpreter drives, hot regions tier up.
        Shape::Threaded => interp_driven(m),
        Shape::Batch { entry } | Shape::Reactor { entry } => {
            // Wasm-drivable iff rooted-eligible AND nothing reachable can suspend across a wasm frame.
            // The concurrency check makes this selection strictly more conservative than the raw
            // `compile_module_reactor` entry (which would emit a suspending cross-tier callee it can't
            // safely unwind) — closing that latent sharp edge.
            if !reachable_concurrency(m, entry) {
                if let Ok((wasm, emitted)) = compile_module_reactor(m, entry, shared_memory) {
                    return Ok(Artifact {
                        wasm,
                        emitted,
                        drive: DriveMode::WasmDriven { entry },
                    });
                }
            }
            interp_driven(m)
        }
    }
}

/// Assemble the wasm module: emit the functions listed in `emitted` (SVM indices, in the order they
/// take wasm indices), routing each `Call` via `wasm_of` (a direct wasm call) or, for an interp
/// leaf, through `env.call_interp`. See the module docs for the emitted shape.
#[allow(clippy::too_many_arguments)]
fn emit_module(
    m: &Module,
    shared_memory: bool,
    emitted: &[usize],
    wasm_of: &[Option<u32>],
    interp_leaf: &[bool],
    reserved_table_log2: Option<u32>,
    nested_caps: bool,
    paged: Option<u8>,
    null_guard: Option<u64>,
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
    // §14 host-bounce import types (added up front so the import section can reference them):
    //   env.instantiate: (i32 win, i32 inst, i64 entry, i64 off, i64 size_log2, i64 quota) -> i32
    //   env.join:        (i32 inst, i32 child) -> i64
    let (mut instantiate_ty, mut join_ty) = (0u32, 0u32);
    let uses_rec = nested_caps && module_uses_rec(m);
    let mut instantiate_rec_ty = 0u32;
    let (mut thread_spawn_ty, mut thread_join_ty, mut mem_wait_ty, mut mem_notify_ty) =
        (0u32, 0u32, 0u32, 0u32);
    if nested_caps {
        instantiate_ty = types.len() as u32;
        types.push((vec![0x7f, 0x7f, 0x7e, 0x7e, 0x7e, 0x7e], vec![0x7f]));
        join_ty = types.len() as u32;
        types.push((vec![0x7f, 0x7f], vec![0x7e]));
        if uses_rec {
            // §3c.3 env.instantiate_rec: (i32 win, i32 inst, i64 record_ptr) -> i32 child handle
            instantiate_rec_ty = types.len() as u32;
            types.push((vec![0x7f, 0x7f, 0x7e], vec![0x7f]));
        }
        // §11 thread/futex bounces:
        //   env.thread_spawn: (i32 func, i64 sp, i64 arg) -> i32 handle
        //   env.thread_join:  (i32 handle) -> i64 result
        //   env.mem_wait:     (i32 win, i64 addr, i64 expected, i64 timeout, i32 is64) -> i32 status
        //   env.mem_notify:   (i32 win, i64 addr, i32 count) -> i32 woken
        thread_spawn_ty = types.len() as u32;
        types.push((vec![0x7f, 0x7e, 0x7e], vec![0x7f]));
        thread_join_ty = types.len() as u32;
        types.push((vec![0x7f], vec![0x7e]));
        mem_wait_ty = types.len() as u32;
        types.push((vec![0x7f, 0x7e, 0x7e, 0x7e, 0x7f], vec![0x7f]));
        mem_notify_ty = types.len() as u32;
        types.push((vec![0x7f, 0x7e, 0x7f], vec![0x7f]));
    }
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
    // The funcref table (`RefFunc`/`dispatch_indirect` semantics): masking `idx & (table_size - 1)`
    // in the lowering reproduces `dispatch_indirect`'s `idx & (len - 1)`.
    //
    // Two shapes, one lowering:
    // * **Local (default, `reserved_table_log2 = None`).** Slot `s` = SVM function `s`, power-of-two
    //   length, trapping (null) padding — matching the interpreter's `DomainTable`
    //   (`funcs.len().next_power_of_two()`, `reserve_log2 = 0`). The module declares its own table and
    //   fills `[0, funcs.len())` with an active element segment (below).
    // * **Shared reserved (Model B2, `reserved_table_log2 = Some(log2)`).** The module *imports* one
    //   shared `env.__indirect_function_table` sized to the domain's reservation (`1 << log2`, exactly
    //   `Host::jit_table_log2` / `DomainTable::new(_, log2)`), and populates *no* slots itself — the
    //   host writes every slot (its own funcs and any `install`ed unit) via `table.set`, exactly as
    //   `DomainTable` is host-populated. So an `install`ed unit becomes a funcref that another
    //   instance's `call_indirect` reaches through the one shared table (§22 old→new). The mask is a
    //   constant `1<<log2` from t=0 (invariant I2), so no compiled site holds a stale mask.
    let table_size = match reserved_table_log2 {
        Some(log2) => 1u32 << log2,
        None => m.funcs.len().next_power_of_two().max(1) as u32,
    };

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
    // Trampolines/trap-stub + the active element that references them only exist for the *local*
    // table the module fills itself. In shared-reserved (B2) mode the host owns every slot, so the
    // module emits no element segment and needs none of these fillers.
    if needs_table && reserved_table_log2.is_none() {
        // **Every** cross-tier function needs a trampoline slot — not just the `RefFunc`
        // address-taken ones. A function pointer can be an indirect-call target without any `RefFunc`
        // instruction: the frontend bakes static function-pointer tables (e.g. Doom's `states[]` /
        // `mobjinfo[]` action functions) into **data segments** as plain function-index constants,
        // invisible to a RefFunc scan. So the identity table must route *any* index to its function,
        // exactly as the interpreter's `DomainTable` does — otherwise a `call_indirect` through a
        // data-segment pointer hits a trap stub ("null function or function signature mismatch") the
        // interpreter would have dispatched. (Fixed a hang/trap ~frame 174 of Doom, when the first
        // monster thinker fires an `A_*` action loaded from `states[]`.)
        let mut next_widx = if nested_caps {
            NESTED_IMPORTED_FUNCS + uses_rec as u32
        } else {
            IMPORTED_FUNCS
        } + emitted.len() as u32;
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
            nested_caps,
            paged,
            null_guard,
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

    // Import section (2): env.memory, env.trap (type 0), env.call_interp (type 1); plus — in
    // shared-reserved (B2) mode — the shared funcref table, and — in §14 `nested_caps` mode —
    // env.instantiate + env.join (the VM-in-VM host bounce).
    let mut sec = Vec::new();
    let n_imports = 3
        + reserved_table_log2.is_some() as u64
        + if nested_caps { 6 } else { 0 }
        + uses_rec as u64;
    uleb(&mut sec, n_imports);
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
    if nested_caps {
        // §14 VM-in-VM host bounces (func imports 2 and 3, see INSTANTIATE_IMPORT_IDX / JOIN_IMPORT_IDX).
        import_name(&mut sec, "env", "instantiate");
        sec.push(0x00); // func
        uleb(&mut sec, instantiate_ty as u64);
        import_name(&mut sec, "env", "join");
        sec.push(0x00); // func
        uleb(&mut sec, join_ty as u64);
        import_name(&mut sec, "env", "thread_spawn");
        sec.push(0x00); // func
        uleb(&mut sec, thread_spawn_ty as u64);
        import_name(&mut sec, "env", "thread_join");
        sec.push(0x00); // func
        uleb(&mut sec, thread_join_ty as u64);
        import_name(&mut sec, "env", "mem_wait");
        sec.push(0x00); // func
        uleb(&mut sec, mem_wait_ty as u64);
        import_name(&mut sec, "env", "mem_notify");
        sec.push(0x00); // func
        uleb(&mut sec, mem_notify_ty as u64);
        if uses_rec {
            // §3c.3 — appended last (func import 8), so no existing import index shifts.
            import_name(&mut sec, "env", "instantiate_rec");
            sec.push(0x00); // func
            uleb(&mut sec, instantiate_rec_ty as u64);
        }
    }
    if reserved_table_log2.is_some() {
        // The shared reserved funcref table (§22 Model B2). Imported (not declared), so every
        // instance in the domain dispatches through the *same* table; the host sizes it to the
        // reservation and populates it (`table.set` on grant + `install`). `table_size == 1<<log2`.
        import_name(&mut sec, "env", "__indirect_function_table");
        sec.push(0x01); // table
        sec.push(0x70); // funcref elemtype
        sec.push(0x00); // limits flag 0x00 = min only
        uleb(&mut sec, table_size as u64);
    }
    section(&mut out, 2, &sec);

    let mut sec = Vec::new(); // function section (3): emitted, then trampolines + trap stub
    uleb(&mut sec, (fn_type_idx.len() + extra_type_idx.len()) as u64);
    for ti in fn_type_idx.iter().chain(&extra_type_idx) {
        uleb(&mut sec, *ti as u64);
    }
    section(&mut out, 3, &sec);

    if needs_table && reserved_table_log2.is_none() {
        let mut sec = Vec::new(); // table section (4): one funcref table, min = table_size
        uleb(&mut sec, 1);
        sec.push(0x70); // funcref elemtype
        sec.push(0x00); // limits flag 0x00 = min only
        uleb(&mut sec, table_size as u64);
        section(&mut out, 4, &sec);
    }

    {
        // Global section (6): the emitted module's mutable globals, in index order.
        //  * global 0 (`FUEL_GLOBAL_IDX`) — the fuel counter, self-initialized to the standard
        //    per-region budget (`FUEL_DEFAULT`) so an unseeded region still runs; the host may re-arm
        //    or tighten it via the exported `"fuel"` global. Debited in `emit_fuel_check`.
        //  * global `MAPPED_GLOBAL_IDX` (1) — the live window `mapped` size (#717), self-
        //    initialized to the emit-time `1 << size_log2`. A host that never grows behaves exactly as
        //    the old baked constant; a `vm_map`-growing host writes the live size via the `"mapped"`
        //    export, and `emit_confine`/`emit_span_check` read it live. Register-allocatable; no guest
        //    store can alias it.
        let mut sec = Vec::new();
        uleb(&mut sec, 2 + paged.is_some() as u64);
        // The fuel global (index 0).
        sec.push(0x7e); // i64
        sec.push(0x01); // mutable
        sec.push(OP_I64_CONST);
        sleb64(&mut sec, FUEL_DEFAULT);
        sec.push(OP_END);
        // The `mapped` global: default = the emit-time window size (`1 << size_log2`).
        sec.push(0x7e); // i64
        sec.push(0x01); // mutable
        sec.push(OP_I64_CONST);
        sleb64(&mut sec, mapped as i64);
        sec.push(OP_END);
        if paged.is_some() {
            // The `pagestate` global (#750, paged modules only): the linear-memory base of the
            // host-maintained byte-per-page state table, written by the driver before each emitted
            // call (entry snapshot — emitted code can never reach a page op, so state is frozen
            // while it runs). Default `0` — an opted-in host that never writes it reads arbitrary
            // in-memory bytes as states, which mis-traps but stays inside the mask (no escape).
            sec.push(0x7f); // i32
            sec.push(0x01); // mutable
            sec.push(OP_I32_CONST);
            sleb64(&mut sec, 0);
            sec.push(OP_END);
        }
        section(&mut out, 6, &sec);
    }

    let mut sec = Vec::new(); // export section (7): "f{svm_idx}" → its wasm index
                              // One `f{i}` per emitted function, plus the `"fuel"` and `"mapped"` globals
                              // (both always — #717 host sync), plus `"pagestate"` on paged modules.
    let n_exports = emitted.len() as u64 + 2 + paged.is_some() as u64;
    uleb(&mut sec, n_exports);
    for &fi in emitted {
        let name = format!("f{fi}");
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x00);
        uleb(&mut sec, wasm_of[fi].unwrap() as u64);
    }
    {
        let name = "fuel";
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x03); // global export kind
        uleb(&mut sec, FUEL_GLOBAL_IDX as u64);
    }
    {
        // The live-`mapped` global (#717): exported so a `vm_map`-growing host can write the live size.
        let name = "mapped";
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x03); // global export kind
        uleb(&mut sec, MAPPED_GLOBAL_IDX as u64);
    }
    if paged.is_some() {
        // The page-state table base (#750, paged modules only): the driver writes it before each
        // emitted call, alongside `"mapped"`.
        let name = "pagestate";
        uleb(&mut sec, name.len() as u64);
        sec.extend_from_slice(name.as_bytes());
        sec.push(0x03); // global export kind
        uleb(&mut sec, (MAPPED_GLOBAL_IDX + 1) as u64);
    }
    section(&mut out, 7, &sec);

    if needs_table && reserved_table_log2.is_none() {
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

    // Name section (custom id 0, "name", function-names subsection) — names each emitted wasm function by
    // its **export symbol name** (`lua_checkstack`, `JS_CallInternal` …), plus the cross-tier trampolines
    // (`xtramp_gfN`) and the `()->()` trap stub (`TRAP_STUB`). Diagnostic-only and cheap: a V8 wasm stack
    // trace — e.g. a `svm_warm_jit` decline (issue #865) — then names the culprit function instead of a
    // bare `wasm-function[N]`. `wasm_of[fi]` is the final wasm index (it already accounts for the function
    // imports), so it is the name-section funcidx directly; `env.trap`/`env.call_interp` are always
    // imports 0/1.
    //
    // Name precedence (#907): a DWARF source name (a `-g` build's `debug_info.func_names`, the reserved
    // level-2 rung) → the function's export symbol name → the guest index `gf{fi}`. Every defined function
    // is exported by its symbol name (translate's `exports` = one entry per defined function), so the real
    // names already ship in every `.svmb` via the semantic exports table — no `-g` flag, no format change,
    // no extra bytes. `gf{fi}` remains only for an *unexported* emitted function (a synthesized helper /
    // trampoline). Built once into maps: a real module's `exports` runs to thousands of entries.
    {
        use std::collections::HashMap;
        let dwarf: HashMap<u32, &str> = m
            .debug_info
            .as_ref()
            .map(|d| {
                d.func_names
                    .iter()
                    .map(|f| (f.func, f.name.as_str()))
                    .collect()
            })
            .unwrap_or_default();
        let export_name: HashMap<u32, &str> = m
            .exports
            .iter()
            .map(|e| (e.func, e.name.as_str()))
            .collect();
        let src_name = |fi: usize| -> String {
            let fi = fi as u32;
            dwarf
                .get(&fi)
                .or_else(|| export_name.get(&fi))
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("gf{fi}"))
        };
        let mut ents: Vec<(u32, String)> = vec![
            (0, "env.trap".to_string()),
            (1, "env.call_interp".to_string()),
        ];
        for fi in 0..m.funcs.len() {
            if let Some(w) = wasm_of[fi] {
                ents.push((w, src_name(fi)));
            }
            if let Some(w) = tramp_of[fi] {
                ents.push((w, format!("xtramp_gf{fi}")));
            }
        }
        if let Some(w) = trap_stub_widx {
            ents.push((w, "TRAP_STUB".to_string()));
        }
        ents.sort_by_key(|(w, _)| *w);
        ents.dedup_by_key(|(w, _)| *w);
        let mut names_sub = Vec::new();
        uleb(&mut names_sub, ents.len() as u64);
        for (w, nm) in &ents {
            uleb(&mut names_sub, *w as u64);
            uleb(&mut names_sub, nm.len() as u64);
            names_sub.extend_from_slice(nm.as_bytes());
        }
        let mut payload = Vec::new();
        uleb(&mut payload, 4);
        payload.extend_from_slice(b"name");
        payload.push(0x01); // subsection 1: function names
        uleb(&mut payload, names_sub.len() as u64);
        payload.extend_from_slice(&names_sub);
        section(&mut out, 0, &payload);
    }

    Ok(out)
}

/// The cross-tier `env.call_interp` slot ABI, **store half**: with a value already on the stack
/// (its scratch-slot address pushed beneath it), emit the widen/reinterpret + `store` that packs it
/// into its scratch slot(s). Paired with [`emit_slot_load`] — the two are the ABI's single encoding,
/// so the emitter and every host servicer agree byte-for-byte. `i32` widens to the full slot; `f32`
/// writes its low 4 bytes (the high 4 stay stale, unread); `i64`/`f64` fill the slot; `v128` fills
/// **two** consecutive slots (16 raw little-endian bytes, #749 — the slot address is only 8-aligned,
/// hence the a=8 alignment hint). Slot addresses come from [`slot_off`], never `i*8`.
fn emit_slot_store(code: &mut Vec<u8>, ty: ValType) {
    match ty {
        ValType::I32 => code.extend_from_slice(&[0xad, 0x37, 0x03, 0x00]), // i64.extend_i32_u; i64.store a=8
        ValType::I64 => code.extend_from_slice(&[0x37, 0x03, 0x00]),       // i64.store a=8
        ValType::F32 => code.extend_from_slice(&[0x38, 0x02, 0x00]),       // f32.store a=4
        ValType::F64 => code.extend_from_slice(&[0x39, 0x03, 0x00]),       // f64.store a=8
        ValType::V128 => code.extend_from_slice(&[0xfd, 0x0b, 0x03, 0x00]), // v128.store a=8
        _ => unreachable!("marshallable_sig admits only i32/i64/f32/f64/v128"),
    }
}

/// The cross-tier `env.call_interp` slot ABI, **load half**: with a scratch-slot address on the
/// stack, emit the `load` (+ narrow) that reads the slot(s) back to `ty`. The inverse of
/// [`emit_slot_store`]; see that function for the encoding.
fn emit_slot_load(code: &mut Vec<u8>, ty: ValType) {
    match ty {
        ValType::I32 => code.extend_from_slice(&[0x29, 0x03, 0x00, 0xa7]), // i64.load a=8; i32.wrap_i64
        ValType::I64 => code.extend_from_slice(&[0x29, 0x03, 0x00]),       // i64.load a=8
        ValType::F32 => code.extend_from_slice(&[0x2a, 0x02, 0x00]),       // f32.load a=4
        ValType::F64 => code.extend_from_slice(&[0x2b, 0x03, 0x00]),       // f64.load a=8
        ValType::V128 => code.extend_from_slice(&[0xfd, 0x00, 0x03, 0x00]), // v128.load a=8
        _ => unreachable!("marshallable_sig admits only i32/i64/f32/f64/v128"),
    }
}

/// Emit a **cross-tier indirect trampoline** body for SVM function `fi` (its `Func` is `f`): a wasm
/// function with the env-prepended signature `(win:i32, env:i32, ...params) -> results` that marshals
/// its params into the env scratch, calls `env.call_interp(fi, args_ptr)`, and returns the result
/// slots — the same sequence [`emit_func`] uses for a cross-tier *direct* call, packaged as a
/// standalone function so a cross-tier function whose **address is taken** can fill its funcref-table
/// slot (an indirect call to it then reaches the interpreter). No locals: params are locals
/// `2..2+nparams`; results are loaded straight onto the operand stack for the return.
fn emit_trampoline(f: &Func, fi: u32) -> Result<Vec<u8>, Error> {
    if slot_count(&f.params).max(slot_count(&f.results)) > XCALL_MAX_SLOTS as u64 {
        return Err(Error::Unsupported("indirect trampoline arity too large"));
    }
    let mut code = Vec::new();
    uleb(&mut code, 0); // no local declarations
    for (i, p) in f.params.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 1); // env
        code.push(OP_I32_CONST);
        sleb32(&mut code, (ENV_SCRATCH_OFF + slot_off(&f.params, i)) as i32);
        code.push(0x6a); // i32.add → slot addr
        code.push(OP_LOCAL_GET);
        uleb(&mut code, (2 + i) as u64); // the i-th SVM param local
        emit_slot_store(&mut code, *p);
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
        sleb32(
            &mut code,
            (ENV_SCRATCH_OFF + slot_off(&f.results, i)) as i32,
        );
        code.push(0x6a); // i32.add
        emit_slot_load(&mut code, *r);
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

/// #846 — a standalone **cross-tier trampoline module**: one exported function `"t"` with the
/// env-prepended `call_indirect` signature of `(params) -> (results)`, whose body marshals its
/// params into the env scratch, calls `env.call_interp(target, args_ptr)`, and returns the
/// reloaded result slots — the [`emit_trampoline`] body packaged as its own instantiable module,
/// so a **host-populated** shared table (§22 Model B2) can route a slot whose occupant has no
/// emitted wasm (a cross-tier program function, an interpreter-only installed unit) back to a
/// **live-state** interpreter bounce. `target` is the value handed to `env.call_interp` — the
/// dispatch-table slot, which for the natural prefix is the program function's own index. The
/// import layout matches the emitted modules' (`env.memory`, `env.trap`, `env.call_interp`), so
/// the one import object a host builds serves emitted units and trampolines alike; the host's
/// `call_interp` signals a trap by throwing (unwinding the emitted frames), exactly the
/// cross-tier convention everywhere. Rejects non-scalar and over-arity signatures — this §22
/// transport's `call_interp` lands in the vCPU's `bounce_call`, whose `&[i64]` slot decode is
/// scalar-only (unlike the module-internal cross-tier ABI, which marshals `v128` two-slot since
/// #749) — so `v128` stays fail-closed here until that transport is widened too.
pub fn emit_slot_trampoline(
    params: &[ValType],
    results: &[ValType],
    target: u32,
    shared_memory: bool,
) -> Result<Vec<u8>, Error> {
    let scalar =
        |t: &ValType| matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64);
    if !params.iter().all(scalar) || !results.iter().all(scalar) {
        return Err(Error::Unsupported("non-scalar trampoline signature"));
    }
    let sig = FuncType {
        params: params.to_vec(),
        results: results.to_vec(),
    };
    let f = Func {
        params: params.to_vec(),
        results: results.to_vec(),
        blocks: Vec::new(),
    };
    let body = emit_trampoline(&f, target)?; // arity-guarded there (XCALL_MAX_SLOTS)

    let mut out = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // \0asm v1

    // Type section (1): t0 = env.trap (i32)->(), t1 = env.call_interp (i32,i32)->(), t2 = the
    // trampoline's env-prepended signature.
    let (tp, tr) = indirect_type_bytes(&sig)?;
    let mut sec = Vec::new();
    uleb(&mut sec, 3);
    sec.extend_from_slice(&[0x60, 0x01, 0x7f, 0x00]); // (i32) -> ()
    sec.extend_from_slice(&[0x60, 0x02, 0x7f, 0x7f, 0x00]); // (i32, i32) -> ()
    sec.push(0x60);
    uleb(&mut sec, tp.len() as u64);
    sec.extend_from_slice(&tp);
    uleb(&mut sec, tr.len() as u64);
    sec.extend_from_slice(&tr);
    section(&mut out, 1, &sec);

    // Import section (2): env.memory (share-matched), env.trap (func 0), env.call_interp (func 1)
    // — [`emit_trampoline`]'s body calls func index 1, so the order must match [`emit_module`]'s.
    let mut sec = Vec::new();
    uleb(&mut sec, 3);
    import_name(&mut sec, "env", "memory");
    sec.push(0x02); // memory
    if shared_memory {
        sec.push(0x03); // shared + has-max
        uleb(&mut sec, 0);
        uleb(&mut sec, 65536);
    } else {
        sec.push(0x00); // min-only
        uleb(&mut sec, 0);
    }
    import_name(&mut sec, "env", "trap");
    sec.push(0x00); // func
    uleb(&mut sec, 0);
    import_name(&mut sec, "env", "call_interp");
    sec.push(0x00); // func
    uleb(&mut sec, 1);
    section(&mut out, 2, &sec);

    // Function (3), export (7: "t" → func index 2), code (10).
    let mut sec = Vec::new();
    uleb(&mut sec, 1);
    uleb(&mut sec, 2); // type t2
    section(&mut out, 3, &sec);

    let mut sec = Vec::new();
    uleb(&mut sec, 1);
    uleb(&mut sec, 1);
    sec.extend_from_slice(b"t");
    sec.push(0x00); // func export
    uleb(&mut sec, 2); // imports 0..=1 are funcs, ours is 2
    section(&mut out, 7, &sec);

    let mut sec = Vec::new();
    uleb(&mut sec, 1);
    uleb(&mut sec, body.len() as u64);
    sec.extend_from_slice(&body);
    section(&mut out, 10, &sec);

    Ok(out)
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
    /// Wasm global index of the live-`mapped` window size (#717). `emit_confine`/`emit_span_check`
    /// read it via `global.get` instead of a baked `1 << size_log2`, so a `vm_map`-grown window no
    /// longer spuriously faults a legitimate access on the JIT. See [`MAPPED_GLOBAL_IDX`].
    mapped_global_idx: u32,
    /// The **gated software page-check** (#750): `Some((page_log2, pagestate_global_idx))` iff this
    /// module was compiled by the opt-in paged entry ([`compile_module_tierup_paged`]). Every
    /// confined access then also consults the host-maintained byte-per-page state table (base in
    /// the exported `"pagestate"` i32 global; `0 = Unmapped`, `1 = Rw`, `2 = Ro`) and traps where
    /// the interpreter's `check_prot` would. `None` (every other entry) emits byte-identical code
    /// to before #750 — the fail-closed default pays nothing.
    page_check: Option<(u8, u32)>,
    /// The **experimental NULL-page guard** ([`compile_module_tierup_nullguard`], a measurement
    /// mode): `Some(guard)` ⇒ every confined access additionally traps when its first byte lands
    /// below `guard` — matching an interpreter whose page map seeds `[0, guard)` `Unmapped`. Never
    /// elided (an in-window proof is an upper bound; it says nothing about the low pages). `None`
    /// everywhere else — byte-identical output.
    null_guard: Option<u64>,
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
    nested_caps: bool,
    paged: Option<u8>,
    null_guard: Option<u64>,
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
        .map(|b| block_value_types(m, b, nested_caps))
        .collect::<Result<_, _>>()?;
    // Pool size per type = the max count of that type in any single block.
    const NTYPES: usize = 7; // I32, I64, F32, F64, V128, Ref, Cap (the ValType variants)
    let type_slot = |t: ValType| -> usize {
        match t {
            ValType::I32 => 0,
            ValType::I64 => 1,
            ValType::F32 => 2,
            ValType::F64 => 3,
            ValType::V128 => 4,
            ValType::Ref => 5,
            ValType::Cap => 6, // §3.5 i32-width handle marker
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
        mapped_global_idx: MAPPED_GLOBAL_IDX,
        // The pagestate global (paged mode only) sits immediately after `mapped`.
        page_check: paged.map(|pl| (pl, MAPPED_GLOBAL_IDX + 1)),
        null_guard,
    };

    let mut code = Vec::new();
    // Copy the SVM params into the entry block's param locals ($next defaults to 0 = entry).
    for (i, _) in f.params.iter().enumerate() {
        code.push(OP_LOCAL_GET);
        uleb(&mut code, 2 + i as u64);
        code.push(OP_LOCAL_SET);
        uleb(&mut code, cx.local_of[0][i] as u64);
    }

    // Fuel unification (INVARIANTS.md #9): charge one fuel for the **function-entry** safepoint,
    // once, before the dispatcher loop — the oracle's per-entry charge (top-level entry and each
    // `call`/`call_indirect`/`return_call` into an emitted function, whose body runs this on entry).
    // The rest of the budget is charged at taken back-edges inside `emit_edge`; forward branches and
    // `return` are free. This matches the oracle safepoint-for-safepoint (not the old coarser
    // once-per-dispatch-iteration debit), so the emitted wasm traps `OutOfFuel` at the *identical*
    // point for any budget, not merely "both eventually trap".
    emit_fuel_check(&mut cx, &mut code);

    // The dispatcher: loop { block .. block { br_table $next } code_0 .. code_{N-1} }.
    code.push(OP_LOOP);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
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
            nested_caps,
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

/// Debit one fuel unit from the fuel counter global and trap `TRAP_OUT_OF_FUEL` when it goes negative.
/// Emitted at the IR-anchored safepoints — once at function entry and once per taken back-edge
/// (`emit_edge`), matching the tree-walk/bytecode/Cranelift oracle exactly (INVARIANTS.md #9), so a run
/// traps `OutOfFuel` at the identical safepoint for any budget.
fn emit_fuel_check(cx: &mut FnCtx, code: &mut Vec<u8>) {
    // global.get FUEL; i64.const 1; i64.sub; tee $fuel; global.set FUEL — the counter lives in a
    // mutable global no guest memory store can alias, so V8 register-allocates it.
    code.push(0x23); // global.get
    uleb(code, FUEL_GLOBAL_IDX as u64);
    code.push(OP_I64_CONST);
    sleb64(code, 1);
    code.push(0x7d); // i64.sub
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.fuel_l as u64);
    code.push(0x24); // global.set
    uleb(code, FUEL_GLOBAL_IDX as u64);
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
/// Slice-3 elision decision for one memory access: is `[addr+offset, addr+offset+width)` **provably**
/// within `[0, mapped)` given the block-local upper bound tracked for the address SSA value? Uses the
/// shared [`svm_ir::bounds`] proof — the same predicate the native JIT uses to elide (there it also
/// drops the mask; here only the bounds-trap branch, keeping the clamp). Fail-closed: an unknown bound
/// ([`UB_TOP`]) or any overflow yields `false` (emit the full check).
fn elide_access(ubs: &[u64], addr: ValIdx, offset: u64, width: u64, mapped: u64) -> bool {
    in_window(ub_at(ubs, addr), offset, width as u32, mapped)
}

fn emit_confine(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    addr_local: u32,
    offset: u64,
    width: u64,
    elide: bool,
    write: bool,
) {
    emit_confine_maybe_aligned(cx, code, addr_local, offset, width, false, elide, write)
}

/// Like [`emit_confine`] but, when `align`, also traps `MemoryFault` on a **misaligned** effective
/// address (`eff % width != 0`) — the natural-alignment requirement §12 atomics carry (the
/// interpreter's `check_align`), which a real hardware atomic would also raise. `width` is a power of
/// two for the atomic types (4 or 8), so `width - 1` is the alignment mask.
///
/// **`elide` — the slice-3 redundant-bounds-check elision.** When the caller has *proven* the access
/// in-window ([`svm_ir::bounds::in_window`] over the address's upper bound), the `eff > mapped - width`
/// bounds-trap branch is redundant and skipped. The **`& MASK` clamp is always emitted** regardless of
/// `elide` (escape safety, INVARIANTS #2): so a *wrong* proof here can only skip a trap the oracle
/// would raise — a trap-parity divergence the interpreter differential catches — never a
/// confinement escape. (The native JIT, which also drops the clamp on proof, is the escape-critical
/// consumer of the same predicate; here we keep the clamp, a strictly safer subset.) The alignment
/// trap is **independent of bounds** and is emitted whenever `align`, elided or not. The elision proof
/// uses the emit-time `mapped` (`elide_access`), a *lower* bound on the live [`MAPPED_GLOBAL_IDX`] size
/// the trap branch actually reads (#717) — the window only grows, so a proven-bounded access stays
/// bounded.
/// One page-state consultation of the #750 software page-check: the state byte of the page holding
/// `(ea_l + delta) & MASK` is loaded from the host-maintained table (base in the `"pagestate"`
/// global) and mismatches trap through the existing [`TRAP_MEMORY_FAULT`] seam — a read of an
/// `Unmapped` page, or a write to anything but `Rw`. Emitted only for paged modules
/// ([`FnCtx::page_check`]); the trap decision happens strictly **inside** the already-masked
/// window, so a wrong table is a trap-parity divergence (INVARIANTS #9), never an escape (#2).
fn emit_page_check_one(cx: &mut FnCtx, code: &mut Vec<u8>, delta: u64, write: bool) {
    let Some((page_log2, ps_gidx)) = cx.page_check else {
        return;
    };
    code.push(OP_LOCAL_GET);
    uleb(code, cx.ea_l as u64);
    if delta != 0 {
        code.push(OP_I64_CONST);
        sleb64(code, delta as i64);
        code.push(0x7c); // i64.add → the access's last byte
    }
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and — same clamp domain as the access itself
    code.push(OP_I64_CONST);
    sleb64(code, page_log2 as i64);
    code.push(0x88); // i64.shr_u → window-relative page index
    code.push(0xa7); // i32.wrap_i64
    code.push(0x23); // global.get pagestate (table base in linear memory)
    uleb(code, ps_gidx as u64);
    code.push(0x6a); // i32.add
    code.push(0x2d); // i32.load8_u → the page's state byte
    uleb(code, 0); // align
    uleb(code, 0); // offset
    if write {
        // A store is admitted only on an `Rw` (1) page.
        code.push(OP_I32_CONST);
        sleb64(code, 1);
        code.push(0x47); // i32.ne
    } else {
        // A load is admitted on anything committed (`Rw`/`Ro`) — only `Unmapped` (0) traps.
        code.push(0x45); // i32.eqz
    }
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
}

/// The experimental **NULL-page guard** check ([`compile_module_tierup_nullguard`], a measurement
/// mode): trap when the access's first byte lands below `guard`. A *bottom* guard needs no
/// last-byte consultation — an access starting at or above `guard` cannot reach down into
/// `[0, guard)` — so this is one compare + never-taken branch, the cheap lowering the NULL-trap
/// design weighs against full paged mode. Masked into the same clamp domain as the access itself
/// (matching [`emit_page_check_one`]'s defensive style). No-op when unguarded.
fn emit_null_guard(cx: &mut FnCtx, code: &mut Vec<u8>) {
    let Some(guard) = cx.null_guard else {
        return;
    };
    code.push(OP_LOCAL_GET);
    uleb(code, cx.ea_l as u64);
    code.push(OP_I64_CONST);
    sleb64(code, MASK as i64);
    code.push(0x83); // i64.and — same clamp domain as the access itself
    code.push(OP_I64_CONST);
    sleb64(code, guard as i64);
    code.push(0x54); // i64.lt_u → first byte below the guard?
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
}

#[allow(clippy::too_many_arguments)]
fn emit_confine_maybe_aligned(
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    addr_local: u32,
    offset: u64,
    width: u64,
    align: bool,
    elide: bool,
    write: bool,
) {
    code.push(OP_LOCAL_GET);
    uleb(code, addr_local as u64);
    code.push(OP_I64_CONST);
    sleb64(code, offset as i64);
    code.push(0x7c); // i64.add → eff (unmasked)
    code.push(OP_LOCAL_TEE);
    uleb(code, cx.ea_l as u64);
    if !elide {
        // eff > live_mapped - width ?  — #717: the bound is the **live** window size, read from the
        // `mapped` global (default = the emit-time `1 << size_log2`) rather than a baked constant, so
        // an access into a `vm_map`-grown region no longer faults on the JIT where the interpreter
        // admits it. `i64.sub` wraps exactly like the old `mapped.wrapping_sub(width)` constant did.
        code.push(0x23); // global.get
        uleb(code, cx.mapped_global_idx as u64);
        code.push(OP_I64_CONST);
        sleb64(code, width as i64);
        code.push(0x7d); // i64.sub → live_mapped - width
        code.push(0x56); // i64.gt_u: eff > live_mapped - width ?
        code.push(OP_IF);
        code.push(BLOCKTYPE_VOID);
        cx.depth += 1;
        emit_trap(code, TRAP_MEMORY_FAULT);
        code.push(OP_END);
        cx.depth -= 1;
    } else {
        // Proven in-window: drop the bounds-trap branch. `ea_l` still holds `eff` for the mask below;
        // pop the value the `local.tee` left on the stack (the un-elided path consumes it in `gt_u`).
        code.push(0x1a); // drop
    }
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
    // #750 (paged modules only): the software page-check — first and, when the access can straddle
    // a page boundary, last touched page, exactly the pages the oracle's `check_prot` walks. An
    // `align`ed access never straddles (the align trap above already fired for a misaligned
    // address, and a `width`-aligned access of power-of-two `width` ≤ page size lies in one page),
    // so only unaligned multi-byte accesses consult the second page. NEVER elided: `elide` proves
    // the access in-window, but page *state* is dynamic, so an in-window proof says nothing about
    // mapped/RW (#750's honest-limits note). No-op when unpaged.
    emit_page_check_one(cx, code, 0, write);
    if width > 1 && !align {
        emit_page_check_one(cx, code, width - 1, write);
    }
    // NULL-page guard (experimental measurement mode) — like the page check, never elided.
    emit_null_guard(cx, code);
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
/// the span `[base, base+len)` lies within `[0, live_mapped)` — matching the interpreter's
/// `confine_span`/`check_prot_span` net behaviour (a span above the live `mapped` is uncommitted →
/// faults), and keeping every accessed byte inside the physical window (never the adjacent linear
/// memory). The bound is the **live** window size read from the [`MAPPED_GLOBAL_IDX`] global (#717),
/// not a baked constant, so a `vm_map`-grown span no longer faults where the interpreter admits it. The
/// check is **overflow-safe**: `base > live_mapped`, then (only once `base <= live_mapped`)
/// `len > live_mapped - base`, so `live_mapped - base` can't underflow and `base + len` can't overflow.
///
/// Called **inside an `if len != 0` guard** (see the lowering in `emit_block_body`), so `len >= 1`
/// here and a passed check guarantees `base < live_mapped` — making [`emit_win_addr`]'s mask a no-op.
/// Emits nothing to the operand stack; call [`emit_win_addr`] afterwards for each span's confined
/// address.
fn emit_span_check(cx: &mut FnCtx, code: &mut Vec<u8>, base_local: u32, len_local: u32) {
    // trap if base > live_mapped  (#717: live window size from the `mapped` global, not a constant)
    code.push(OP_LOCAL_GET);
    uleb(code, base_local as u64);
    code.push(0x23); // global.get
    uleb(code, cx.mapped_global_idx as u64);
    code.push(0x56); // i64.gt_u
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    // trap if len > live_mapped - base
    code.push(OP_LOCAL_GET);
    uleb(code, len_local as u64);
    code.push(0x23); // global.get
    uleb(code, cx.mapped_global_idx as u64);
    code.push(OP_LOCAL_GET);
    uleb(code, base_local as u64);
    code.push(0x7d); // i64.sub → live_mapped - base (base <= live_mapped here)
    code.push(0x56); // i64.gt_u: len > live_mapped - base
    code.push(OP_IF);
    code.push(BLOCKTYPE_VOID);
    cx.depth += 1;
    emit_trap(code, TRAP_MEMORY_FAULT);
    code.push(OP_END);
    cx.depth -= 1;
    // #1004: NULL-guard low bound for a marked module — trap if the span dips into the reserved
    // `[0, guard)` region. Called inside `if len != 0`, so `len >= 1` and `base` is the span's
    // lowest byte: `base >= guard` proves every byte is at or above the guard (a *bottom* region
    // needs no last-byte check, #750's note). This is the span analogue of the scalar
    // [`emit_null_guard`] — with it, bulk-memory functions of a marked module emit (they no longer
    // leave the subset in `analyze` / the tier-up fixpoint), trapping exactly where the
    // interpreter's `check_prot_span` faults on the `Unmapped` guard pages.
    if let Some(guard) = cx.null_guard {
        code.push(OP_LOCAL_GET);
        uleb(code, base_local as u64);
        code.push(OP_I64_CONST);
        sleb64(code, MASK as i64);
        code.push(0x83); // i64.and — same clamp domain as the access itself
        code.push(OP_I64_CONST);
        sleb64(code, guard as i64);
        code.push(0x54); // i64.lt_u → base below the guard?
        code.push(OP_IF);
        code.push(BLOCKTYPE_VOID);
        cx.depth += 1;
        emit_trap(code, TRAP_MEMORY_FAULT);
        code.push(OP_END);
        cx.depth -= 1;
    }
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
    cx: &mut FnCtx,
    code: &mut Vec<u8>,
    from_block: usize,
    target: u32,
    args: &[svm_ir::ValIdx],
) {
    // Fuel unification (INVARIANTS.md #9): charge one fuel per **taken back-edge** — a branch whose
    // target block index is `<= from_block`, the oracle's exact definition (svm-interp `drive`).
    // Both indices are static per edge, so the charge is unconditional here (Br) or lands inside the
    // taken arm (BrIf/BrTable's per-target `emit_edge`); forward branches are free. Stack-neutral, so
    // it precedes the arg shuffle cleanly.
    if (target as usize) <= from_block {
        emit_fuel_check(cx, code);
    }
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
    nested_caps: bool,
) -> Result<(), Error> {
    let mut next_val = b.params.len(); // where the next instruction's results land
                                       // Slice-3 confinement-check elision: a block-local **upper-bound** map over SSA values, in
                                       // lockstep with the value numbering `next_val` drives (block params carry no bound → `UB_TOP`;
                                       // bounds do not cross block boundaries). At each memory access, `elide_access` consults the
                                       // address value's bound via the shared `svm_ir::bounds` proof to decide whether the runtime
                                       // bounds check is redundant. Kept aligned to `next_val` by construction below — a misalignment
                                       // could only mis-elide (a trap-parity divergence the differential catches; never an escape,
                                       // since the `& MASK` clamp is always emitted).
    let mut ubs: Vec<u64> = vec![UB_TOP; b.params.len()];
    let get = |code: &mut Vec<u8>, cx: &FnCtx, v: svm_ir::ValIdx| {
        code.push(OP_LOCAL_GET);
        uleb(code, cx.local_of[k][v as usize] as u64);
    };
    for inst in &b.insts {
        let val_before = next_val;
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
            // Standalone fence: a pure ordering barrier with no data effect. The wasm-JIT only
            // compiles single-threaded guests (concurrency folds to the interp), where a fence is
            // observably a no-op — so emit nothing (matching the oracle, which honors it identically
            // single-threaded).
            Inst::AtomicFence { .. } => {}
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
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    false, // read
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
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    true, // write
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
                    true,
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    false, // read
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
                    true,
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    true, // write
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
                    true,
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    true, // write (RMW/cmpxchg may store)
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
                    true,
                    elide_access(&ubs, *addr, *offset, width, mapped),
                    true, // write (RMW/cmpxchg may store)
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
                emit_span_check(cx, code, dl, ll);
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
                emit_span_check(cx, code, dl, ll);
                emit_span_check(cx, code, sl, ll);
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
            // §11 thread/futex bounces (opt-in `nested_caps`, CONSOLIDATION.md §11 slice 3): the
            // four ops marshal their operands to host imports; the servicer supplies the unit's
            // module (it knows which module it instantiated), so a spawn from an emitted unit is the
            // same module-aware spawn the interpreter does. `thread_join`/`mem_wait` block the
            // calling thread — the Worker/par-tier model (each vCPU is a real thread).
            Inst::ThreadSpawn { func, sp, arg } if nested_caps => {
                code.push(OP_I32_CONST);
                sleb32(code, *func as i32);
                get(code, cx, *sp);
                get(code, cx, *arg);
                code.push(OP_CALL);
                uleb(code, THREAD_SPAWN_IMPORT_IDX as u64);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::ThreadJoin { handle } if nested_caps => {
                get(code, cx, *handle);
                code.push(OP_CALL);
                uleb(code, THREAD_JOIN_IMPORT_IDX as u64);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::MemoryWait {
                ty,
                addr,
                expected,
                timeout,
            } if nested_caps => {
                code.push(OP_LOCAL_GET);
                uleb(code, 0); // win — the servicer confines addr into the window
                get(code, cx, *addr);
                get(code, cx, *expected);
                if matches!(ty, IntTy::I32) {
                    code.push(0xad); // i64.extend_i32_u — uniform i64 expected slot
                }
                get(code, cx, *timeout);
                code.push(OP_I32_CONST);
                sleb32(code, matches!(ty, IntTy::I64) as i32);
                code.push(OP_CALL);
                uleb(code, MEM_WAIT_IMPORT_IDX as u64);
                set_result(cx, code, k, &mut next_val);
            }
            Inst::MemoryNotify { addr, count } if nested_caps => {
                code.push(OP_LOCAL_GET);
                uleb(code, 0); // win
                get(code, cx, *addr);
                get(code, cx, *count);
                code.push(OP_CALL);
                uleb(code, MEM_NOTIFY_IMPORT_IDX as u64);
                set_result(cx, code, k, &mut next_val);
            }
            // §14 VM-in-VM bounce (opt-in `nested_caps`): a `cap.call` to INSTANTIATOR
            // `instantiate`/`join` marshals its operands to a host import (`env.instantiate` /
            // `env.join`, the funcref-table-free analog of `env.call_interp`) that spawns/joins the
            // confined child vCPU host-side — exactly as the interpreter surfaces `VcpuStop::Instantiate`
            // to its driver. The emitted parent does no confinement itself; the child's window carve +
            // attenuated powerbox are the host's job (unchanged from the interpreter path).
            Inst::CapCall {
                type_id,
                op,
                sig,
                handle,
                args,
            } if nested_caps && is_nested_cap(*type_id, *op) => {
                let n_results = sig.results.len();
                if *op == 0 {
                    // env.instantiate(win, inst, entry, off, size_log2, quota) -> i32 child handle
                    code.push(OP_LOCAL_GET);
                    uleb(code, 0); // win — the carve base is window-relative
                    get(code, cx, *handle); // the Instantiator capability handle (i32)
                    for a in args {
                        get(code, cx, *a); // entry, off, size_log2, quota
                    }
                    code.push(OP_CALL);
                    uleb(code, INSTANTIATE_IMPORT_IDX as u64);
                } else if *op == 17 {
                    // §3c.3 env.instantiate_rec(win, inst, record_ptr) -> i32 child handle — the
                    // config-record spawn; the servicer reads + validates the 56-byte record from
                    // linear memory at win + record_ptr (window-relative, like every §14 pointer).
                    code.push(OP_LOCAL_GET);
                    uleb(code, 0); // win
                    get(code, cx, *handle);
                    for a in args {
                        get(code, cx, *a); // record_ptr
                    }
                    code.push(OP_CALL);
                    uleb(code, INSTANTIATE_REC_IMPORT_IDX as u64);
                } else {
                    // env.join(inst, child) -> i64 result
                    get(code, cx, *handle); // the Instantiator handle
                    for a in args {
                        get(code, cx, *a); // the child handle
                    }
                    code.push(OP_CALL);
                    uleb(code, JOIN_IMPORT_IDX as u64);
                }
                for i in (0..n_results).rev() {
                    code.push(OP_LOCAL_SET);
                    uleb(code, cx.local_of[k][next_val + i] as u64);
                }
                next_val += n_results;
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
                    // Cross-tier: `func` is an interp leaf. Marshal args into the env scratch
                    // slots, call `env.call_interp(func, args_ptr)` (the engine runs it on the
                    // bytecode interpreter and writes results back to the same slots), then reload.
                    None => {
                        if !interp_leaf[*func as usize] {
                            return Err(Error::Unsupported("call to a non-emitted, non-leaf func"));
                        }
                        if slot_count(&callee.params).max(slot_count(&callee.results))
                            > XCALL_MAX_SLOTS as u64
                        {
                            return Err(Error::Unsupported("cross-tier call arity too large"));
                        }
                        // Store each arg to env + ENV_SCRATCH_OFF + slot_off (widen/reinterpret).
                        for (i, a) in args.iter().enumerate() {
                            code.push(OP_LOCAL_GET);
                            uleb(code, 1); // env
                            code.push(OP_I32_CONST);
                            sleb32(code, (ENV_SCRATCH_OFF + slot_off(&callee.params, i)) as i32);
                            code.push(0x6a); // i32.add → slot addr
                            get(code, cx, *a);
                            emit_slot_store(code, callee.params[i]);
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
                            sleb32(
                                code,
                                (ENV_SCRATCH_OFF + slot_off(&callee.results, i)) as i32,
                            );
                            code.push(0x6a); // i32.add
                            emit_slot_load(code, callee.results[i]);
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
                    elide_access(&ubs, *addr, *offset, 16, mapped),
                    false, // read
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
                    elide_access(&ubs, *addr, *offset, 16, mapped),
                    true, // write
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
            _ => return Err(Error::Unsupported("instruction outside the v1 subset")),
        }
        // Keep `ubs` in lockstep with the value numbering `next_val` drives. An instruction that
        // produced exactly one result gets its (possibly-`UB_TOP`) modeled bound; zero- or
        // multi-result instructions (stores, fences, calls, cmpxchg pairs) get `UB_TOP` per new
        // value. `ub_of` reads only operands (values defined earlier, already in `ubs`).
        if next_val == val_before + 1 {
            let bound = ub_of(inst, &ubs);
            ubs.push(bound);
        } else {
            ubs.resize(next_val, UB_TOP);
        }
    }
    debug_assert_eq!(next_val, value_types.len());
    debug_assert_eq!(
        ubs.len(),
        value_types.len(),
        "ubs must stay in lockstep with values"
    );

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
                    if slot_count(&callee.params).max(slot_count(&callee.results))
                        > XCALL_MAX_SLOTS as u64
                    {
                        return Err(Error::Unsupported("cross-tier tail-call arity too large"));
                    }
                    for (i, a) in args.iter().enumerate() {
                        code.push(OP_LOCAL_GET);
                        uleb(code, 1); // env
                        code.push(OP_I32_CONST);
                        sleb32(code, (ENV_SCRATCH_OFF + slot_off(&callee.params, i)) as i32);
                        code.push(0x6a); // i32.add → slot addr
                        get(code, cx, *a);
                        emit_slot_store(code, callee.params[i]);
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
                        sleb32(
                            code,
                            (ENV_SCRATCH_OFF + slot_off(&callee.results, i)) as i32,
                        );
                        code.push(0x6a); // i32.add
                        emit_slot_load(code, callee.results[i]);
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
