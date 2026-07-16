#![forbid(unsafe_code)]
#![cfg_attr(not(test), no_std)]
//! `svm-opt` — the generic svm-IR AOT optimizer (see `OPT.md`, `DESIGN.md` §20a/§20c).
//!
//! A pure, closed-module `Module -> Module` transform: it rewrites verified IR into faster/smaller
//! verified IR, and its output is **re-verified** (`svm_verify::verify_module`) before it runs, so a
//! bug here is a clean verify error, never an escape (untrusted-for-escape posture, §20a). This crate
//! is also the home of the shared **constant-fold machinery** ([`Known`] + the `fold_*` helpers),
//! which the specializer in `svm-peval` (the first Futamura projection) reuses — `svm-peval` depends
//! on this crate and re-exports [`optimize_module`] for backward compatibility.
//!
//! The optimizer is a closed-module, semantics-preserving pass that proves the
//! `rewrite → re-verify → run` loop end to end. It iterates the following to a fixpoint:
//!
//! 1. **Constant folding.** A pure, single-result integer op whose operands are all known
//!    constants is replaced *in place* with the equivalent `const`. Because the
//!    replacement has the same result arity (1), block-local value indices are untouched,
//!    so every downstream operand stays valid with zero renumbering. Folding matches the
//!    reference interpreter's arithmetic **exactly** (`bin32`/`bin64`/`cmp*`/`intun*`/
//!    conversions), and an op that *would trap* (div/rem by zero, signed `INT_MIN/-1`) is
//!    deliberately **left alone** so the residual traps identically.
//! 2. **Branch resolution.** A `br_if`/`br_table` whose selector folded to a constant is
//!    rewritten to an unconditional `br` to the taken edge — using the interpreter's own
//!    selection rule (`cond != 0`; `targets[idx as u32] else default`).
//! 3. **Dead-block elimination.** After branch resolution, blocks unreachable from the
//!    entry are dropped and the remaining blocks renumbered (terminator targets remapped).
//! 4. **Dead-value elimination (Stage 0.x).** Within each block, an instruction that is
//!    pure *and* cannot trap *and* has no side effect (see [`is_removable_if_dead`]) is
//!    removed when none of its results are used by a live instruction or the terminator.
//!    This is the transform that makes folding *pay off* — once a `br_if` resolves, the
//!    code that computed its condition becomes dead and disappears. Removing an instruction
//!    shifts every later block-local value index, so this is the one transform that
//!    **renumbers values**: it relies on the exhaustive operand remapper ([`map_operands`]
//!    / [`map_term_operands`]) to rewrite every surviving operand. Conservatism is by
//!    design — anything that can fault (loads, atomics, trapping conversions) or has an
//!    effect (stores, calls, fences, fiber/thread ops) is *kept* even if its result is
//!    unused, so trap and effect behavior is identical to the source.
//! 5. **Block merging.** A block reached by exactly one edge — an unconditional `br` from its
//!    sole predecessor — is fused into that predecessor (its parameters bind to the branch
//!    arguments). This collapses the `br`-chains the specializer emits into straight-line code.
//! 6. **Dead block-parameter elimination.** A block parameter never referenced in its block is
//!    dropped, along with the matching argument in every predecessor edge — a cross-edge
//!    transform, paired with merging so residuals don't carry threaded-through dead state.
//! 7. **Copy propagation + algebraic identities.** Within a block, a value that is a *copy* of an
//!    earlier value — a constant-condition `select`, or an identity (`x+0`/`x-0`/`x*1`/`x<<0`,
//!    `x|0`/`x&-1`/`x^0`, `x&x`/`x|x`) — has its uses rewritten to that earlier value, so the copy
//!    becomes dead for step 4. Absorbing / self-cancelling forms that yield a *constant* even with
//!    one operand unknown (`x*0`/`x&0` → 0, `x|-1` → -1, `x-x`/`x^x` → 0, `x%1` → 0) fold in step 1.
//!
//! **Untrusted for escape (§2a / §20a posture).** Like the LLVM on-ramp, this pass is
//! *not* in the escape-TCB: its output is meant to be re-verified with
//! `svm_verify::verify_module` before it runs, so a bug here is a clean verify error, never
//! an escape. The differential harness (`tests/optimize.rs`) is the correctness spec:
//! `optimized(args) == original(args)` on the reference interpreter, for results *and*
//! traps — including a randomized fuzz over dead-heavy arithmetic DAGs that stresses the
//! renumbering/remapper.
//!
//! **Float and v128 (SIMD) constant folding** are done — `f32`/`f64` arithmetic / compares / FMA /
//! conversions / casts, and **every** SIMD lane op folds bit-for-bit the interpreter: the common ones
//! (splat, extract/replace, lane int+float arithmetic / compares / shifts, bitwise, shuffle, swizzle)
//! and the exotic ones (saturating add/sub, widen/narrow, lane convert, dot, pairwise, pmin/pmax,
//! avgr, popcnt, any/all-true, bitmask, q15). Float lanes reuse the scalar folds, so NaN/rounding
//! fidelity carries over. The specialization layers that build on this fold machinery (cross-function
//! `call` inlining, narrow renameable cells, value-stack renaming) live in `svm-peval`.
//!
//! **`no_std` + `alloc`.** This crate compiles `no_std` (gated on `not(test)`; its own test harness
//! gets `std`) so it can itself be translated to svm-IR through the Rust on-ramp and run *inside* svm
//! (DESIGN.md §20c). The transform is a pure `Module -> Module` — no I/O, threads, or time — so
//! `core + alloc` suffices; the `std`-only float methods (`sqrt`/`ceil`/`floor`/`trunc`/round-ties-even
//! /`fma`) route through the `libm` crate, which is bit-identical for these correctly-rounded ops.

extern crate alloc;
use alloc::vec; // the `vec!` macro
use alloc::vec::Vec;

use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, IToF, Inst,
    IntTy, IntUnOp, Module, Terminator, VBitBinOp, VCvtOp, VFCmpOp, VFloatBinOp, VFloatUnOp,
    VICmpOp, VIntBinOp, VIntUnOp, VNarrowOp, VPMinMaxOp, VSatBinOp, VShape, VShiftOp, VWidenOp,
    ValIdx, ValType,
};

pub mod cfg;
pub mod gvn;
pub mod instrument;
pub mod licm;
pub mod reassociate;
pub mod sccp;
pub mod ssa;
mod thread;
pub mod vn;

/// A value known to be a constant at optimization time. Tracks scalar integers/floats and `v128`.
/// Floats and `v128` are held as **raw bits/bytes** so equality/hashing are exact and NaN-safe
/// (needed for the specializer's memo key) and folds preserve NaN payloads. Shared with the
/// [`specialize`] engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Known {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    V128([u8; 16]),
}

impl Known {
    /// The `const` instruction that materializes this value (the in-place fold result).
    pub fn to_const_inst(self) -> Inst {
        match self {
            Known::I32(v) => Inst::ConstI32(v),
            Known::I64(v) => Inst::ConstI64(v),
            Known::F32(b) => Inst::ConstF32(b),
            Known::F64(b) => Inst::ConstF64(b),
            Known::V128(b) => Inst::ConstV128(b),
        }
    }
    /// The raw `v128` bytes, if this is one.
    pub fn as_v128(self) -> Option<[u8; 16]> {
        match self {
            Known::V128(b) => Some(b),
            _ => None,
        }
    }
    /// A scalar's low lane bits — the value a `splat`/`replace_lane` writes into a lane.
    pub fn lane_bits(self) -> u64 {
        match self {
            Known::I32(v) => v as u32 as u64,
            Known::I64(v) => v as u64,
            Known::F32(b) => b as u64,
            Known::F64(b) => b,
            Known::V128(_) => 0, // not a scalar; unreachable on a verified module
        }
    }
    pub fn as_i32(self) -> Option<i32> {
        match self {
            Known::I32(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Known::I64(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_f32(self) -> Option<f32> {
        match self {
            Known::F32(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Known::F64(b) => Some(f64::from_bits(b)),
            _ => None,
        }
    }
}

/// Optimize every function in a module. Memory/data/imports/exports are carried through unchanged
/// (optimization is per-function and order-preserving, so funcidxs — and the names that point at
/// them — stay valid); `debug_info` is **dropped** because its `(func, block, inst)` positions go
/// stale once we fold instructions and drop blocks (it is strippable and untrusted for escape, §3a).
pub fn optimize_module(m: &Module) -> Module {
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let has_memory = m.memory.is_some();
    Module {
        funcs: m
            .funcs
            .iter()
            .map(|f| {
                // Global value numbering (OPT.md Phase 2) runs before the per-function cleanup: it
                // eliminates cross-block redundant pure computations (threading the dominating value
                // through block params), then `optimize_func` (SCCP + fixpoint) DCEs the dead
                // duplicates and drops any parameter left unused.
                let g = gvn::gvn(f, &m.funcs, has_memory);
                // Loop-invariant code motion (OPT.md Phase 2): hoist pure, non-trapping invariants
                // out of loops; the fixpoint below DCEs the emptied-out originals.
                let g = licm::licm(&g, &m.funcs, has_memory);
                optimize_func(&g, &fn_results)
            })
            .collect(),
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports: m.exports.clone(),
        debug_info: None,
    }
}

/// Optimize a single function to a fixpoint: fold + resolve branches, prune dead blocks, merge
/// straight-line chains, drop dead block parameters, and drop dead values — repeating until
/// nothing changes. Every pass only simplifies, so this terminates; the cap guards pathologies.
pub fn optimize_func(f: &Func, fn_results: &[usize]) -> Func {
    // SCCP (OPT.md Phase 2) runs first: it propagates constants *globally* — through block
    // parameters and around loops, with conditional reachability — folding what the per-block passes
    // below cannot see. It materializes constants and resolves constant branches; the fixpoint then
    // prunes the newly-unreachable blocks, DCEs the dead selector code, merges, and re-folds.
    let f = sccp::sccp(f, fn_results);
    // Constant reassociation (OPT.md Phase 2): fold `(x OP c1) OP c2` chains so the fixpoint below
    // then folds the combined constants and DCEs the dead inner ops.
    let f = reassociate::reassociate(&f, fn_results);
    let mut blocks: Vec<Block> = f.blocks.iter().map(|b| fold_block(b, fn_results)).collect();
    for _ in 0..1000 {
        let before = blocks.clone();
        blocks = prune_unreachable(blocks);
        blocks = merge_blocks(blocks, fn_results);
        blocks = drop_dead_params(blocks, fn_results);
        // Copy propagation + identity forwarding: rewrite uses of a value that is a copy of an
        // earlier one (a constant-condition `select`, or an algebraic identity like `x+0`/`x*1`)
        // to that earlier value, so the copy instruction becomes dead for the DCE pass below.
        blocks = blocks
            .iter()
            .map(|b| copy_propagate(b, fn_results))
            .collect();
        // Common-subexpression elimination: dedup redundant *pure* computations within a block, so
        // the duplicates become dead for the DCE below.
        blocks = blocks.iter().map(|b| local_cse(b, fn_results)).collect();
        blocks = blocks.iter().map(|b| dce_block(b, fn_results)).collect();
        // Re-fold: merging brings a constant's definition into the same block as its use, and
        // dropping params can expose new constants — both newly foldable here.
        blocks = blocks.iter().map(|b| fold_block(b, fn_results)).collect();
        // Jump threading: redirect an edge that reaches an empty conditional forwarder with a
        // constant selector straight to the resolved target (correlated branches). The next
        // iteration's prune/merge cleans up any forwarder left with no predecessors.
        blocks = jump_thread(&blocks, fn_results);
        if blocks == before {
            break;
        }
    }
    Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks,
    }
}

/// Forward pass over one block: replace foldable instructions with constants in place, then
/// resolve the terminator against the constants discovered. No value indices move.
fn fold_block(b: &Block, fn_results: &[usize]) -> Block {
    // `known[i]` is the constant value (if any) of block-local value index `i`. Seed with the
    // block params (always unknown), then extend by each instruction's result arity in order.
    let mut known: Vec<Option<Known>> = vec![None; b.params.len()];
    let mut insts = b.insts.clone();

    for inst in insts.iter_mut() {
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            if let Some(k) = try_fold(inst, &known) {
                *inst = k.to_const_inst();
                known.push(Some(k));
            } else {
                known.push(const_value(inst));
            }
        } else {
            for _ in 0..rc {
                known.push(None);
            }
        }
    }

    Block {
        params: b.params.clone(),
        insts,
        term: resolve_term(&b.term, &known),
    }
}

/// The constant an instruction *is*, if it is a literal `const` (after folding, every folded
/// op has become one of these). Other instructions carry no statically-known value.
pub(crate) fn const_value(inst: &Inst) -> Option<Known> {
    match *inst {
        Inst::ConstI32(v) => Some(Known::I32(v)),
        Inst::ConstI64(v) => Some(Known::I64(v)),
        Inst::ConstF32(b) => Some(Known::F32(b)),
        Inst::ConstF64(b) => Some(Known::F64(b)),
        Inst::ConstV128(b) => Some(Known::V128(b)),
        _ => None,
    }
}

/// Read a block-local value as a known constant, if it is one.
pub(crate) fn get(known: &[Option<Known>], idx: ValIdx) -> Option<Known> {
    known.get(idx as usize).copied().flatten()
}

/// Try to fold a pure, single-result integer instruction to a constant. Returns `None` when
/// an operand is not known, the op is not foldable, or folding it would trap (div/rem by
/// zero or signed overflow) — in which case the original instruction is kept so the residual
/// traps identically to the source.
pub(crate) fn try_fold(inst: &Inst, known: &[Option<Known>]) -> Option<Known> {
    match *inst {
        Inst::IntBin { ty, op, a, b } => {
            // Both operands known: the exact arithmetic fold.
            if let (Some(x), Some(y)) = (get(known, a), get(known, b)) {
                if let Some(k) = fold_int_bin(ty, op, x, y) {
                    return Some(k);
                }
            }
            // Absorbing-element / self-cancelling identities that yield a *constant* with only one
            // operand known (or `a == b`): `x*0`/`x&0` → 0, `x|-1` → -1, `x-x`/`x^x` → 0, `x%1` → 0.
            fold_absorbing(ty, op, a, b, known)
        }
        Inst::IntCmp { ty, op, a, b } => {
            // Self-comparison folds without knowing the value: `x == x`/`x <= x`/`x >= x` are 1,
            // `x != x`/`x < x`/`x > x` are 0. Integer only — this is unsound for floats (NaN), but
            // float compares are `FCmp`, never here. (The other self-ops — `x-x`, `x^x`, `x&x`,
            // `x|x` — are handled by `fold_absorbing` / `forward_to_operand`.)
            if a == b {
                let is_true = matches!(
                    op,
                    CmpOp::Eq | CmpOp::LeS | CmpOp::LeU | CmpOp::GeS | CmpOp::GeU
                );
                return Some(Known::I32(is_true as i32));
            }
            fold_int_cmp(ty, op, get(known, a)?, get(known, b)?)
        }
        Inst::IntUn { ty, op, a } => fold_int_un(ty, op, get(known, a)?),
        Inst::Eqz { ty, a } => {
            let zero = match ty {
                IntTy::I32 => get(known, a)?.as_i32()? == 0,
                IntTy::I64 => get(known, a)?.as_i64()? == 0,
            };
            Some(Known::I32(zero as i32))
        }
        Inst::Convert { op, a } => match op {
            ConvOp::ExtendI32S => Some(Known::I64(get(known, a)?.as_i32()? as i64)),
            ConvOp::ExtendI32U => Some(Known::I64(get(known, a)?.as_i32()? as u32 as i64)),
            ConvOp::WrapI64 => Some(Known::I32(get(known, a)?.as_i64()? as i32)),
        },
        // `select` with a constant condition folds only when the chosen operand is *itself*
        // a known constant (Stage 0 has no copy/forward op to splice a non-constant through).
        Inst::Select { cond, a, b } => {
            let chosen = if get(known, cond)?.as_i32()? != 0 {
                a
            } else {
                b
            };
            get(known, chosen)
        }
        // Scalar float folds — bit-for-bit the interpreter's `fbin*`/`fun*`/`fcmp*`/`fto_i`/
        // `i_to_f`/`cast`. `FToITrap` folds only when in range (else it is kept so it still traps).
        Inst::FBin { ty, op, a, b } => fold_fbin(ty, op, get(known, a)?, get(known, b)?),
        Inst::FUn { ty, op, a } => fold_fun(ty, op, get(known, a)?),
        Inst::FCmp { ty, op, a, b } => fold_fcmp(ty, op, get(known, a)?, get(known, b)?),
        Inst::Fma { ty, a, b, c } => fold_fma(ty, get(known, a)?, get(known, b)?, get(known, c)?),
        Inst::FToISat { op, a } => fold_ftoi_sat(op, get(known, a)?),
        Inst::FToITrap { op, a } => fold_ftoi_trap(op, get(known, a)?),
        Inst::IToFConv { op, a } => fold_itof(op, get(known, a)?),
        Inst::Cast { op, a } => fold_cast(op, get(known, a)?),
        // v128 (SIMD) lane folds.
        _ => fold_simd(inst, |i| get(known, i)),
    }
}

// ----- scalar float constant folding (mirrors `svm-interp`'s scalar helpers exactly) -----
//
// Floats flow through as raw bits and the math is done in `f32`/`f64`, so NaN payloads and the
// wasm-specific min/max/nearest rules are preserved. The differential harness (interp vs residual,
// on both backends, with NaN/±0/±inf/tie inputs) is the spec.

/// Evaluate a `libm`-only float fold. With the `libm-floats` feature the expression runs; without it
/// (the in-svm svm-IR build, which can't translate libm's inline-asm/i128) the op is never reached —
/// the fold dispatch returns `None` first ([`op_needs_libm`]) — so the arm is `unreachable!`.
macro_rules! libm_only {
    ($e:expr) => {{
        #[cfg(feature = "libm-floats")]
        {
            $e
        }
        #[cfg(not(feature = "libm-floats"))]
        {
            unreachable!("libm float fold reached without the `libm-floats` feature")
        }
    }};
}

/// The unary float ops that need `libm` (no `core` impl): `sqrt`/`ceil`/`floor`/`trunc`/round-ties-
/// even. `abs`/`neg` are pure `core`. The in-svm build (no `libm-floats`) leaves these unfolded.
fn fun_needs_libm(op: FUnOp) -> bool {
    matches!(
        op,
        FUnOp::Sqrt | FUnOp::Ceil | FUnOp::Floor | FUnOp::Trunc | FUnOp::Nearest
    )
}

/// `|a|` in pure `core` (clear the sign bit) — bit-identical to `libm::fabs`/`f*::abs`, so it folds in
/// every build (no libm).
fn fabs32(a: f32) -> f32 {
    f32::from_bits(a.to_bits() & 0x7fff_ffff)
}
fn fabs64(a: f64) -> f64 {
    f64::from_bits(a.to_bits() & 0x7fff_ffff_ffff_ffff)
}
/// `copysign(a, b)` in pure `core` (a's magnitude, b's sign) — bit-identical to `libm::copysign`.
fn fcopysign32(a: f32, b: f32) -> f32 {
    f32::from_bits((a.to_bits() & 0x7fff_ffff) | (b.to_bits() & 0x8000_0000))
}
fn fcopysign64(a: f64, b: f64) -> f64 {
    f64::from_bits((a.to_bits() & 0x7fff_ffff_ffff_ffff) | (b.to_bits() & 0x8000_0000_0000_0000))
}

fn fbin32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin32(a, b),
        FBinOp::Max => fmax32(a, b),
        FBinOp::Copysign => fcopysign32(a, b),
    }
}
fn fbin64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin64(a, b),
        FBinOp::Max => fmax64(a, b),
        FBinOp::Copysign => fcopysign64(a, b),
    }
}
fn fun32(op: FUnOp, a: f32) -> f32 {
    match op {
        // `abs`/`neg` are pure `core` and fold in every build. `sqrt`/`ceil`/`floor`/`trunc`/round-
        // ties-even are not in `core` (std-only) — folded via `libm` only with the `libm-floats`
        // feature; the in-svm build leaves them unfolded (the dispatch returns `None` first).
        FUnOp::Abs => fabs32(a),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => libm_only!(libm::sqrtf(a)),
        FUnOp::Ceil => libm_only!(libm::ceilf(a)),
        FUnOp::Floor => libm_only!(libm::floorf(a)),
        FUnOp::Trunc => libm_only!(libm::truncf(a)),
        FUnOp::Nearest => libm_only!(libm::rintf(a)), // default FP env: round to nearest, ties to even
    }
}
fn fun64(op: FUnOp, a: f64) -> f64 {
    match op {
        FUnOp::Abs => fabs64(a),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => libm_only!(libm::sqrt(a)),
        FUnOp::Ceil => libm_only!(libm::ceil(a)),
        FUnOp::Floor => libm_only!(libm::floor(a)),
        FUnOp::Trunc => libm_only!(libm::trunc(a)),
        FUnOp::Nearest => libm_only!(libm::rint(a)), // default FP env: round to nearest, ties to even
    }
}
// wasm min/max: NaN propagates; for ±0, min prefers -0 and max prefers +0.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
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
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
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
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

/// Fold a binary float op; operands and result are `ty`.
pub fn fold_fbin(ty: FloatTy, op: FBinOp, a: Known, b: Known) -> Option<Known> {
    Some(match ty {
        FloatTy::F32 => Known::F32(fbin32(op, a.as_f32()?, b.as_f32()?).to_bits()),
        FloatTy::F64 => Known::F64(fbin64(op, a.as_f64()?, b.as_f64()?).to_bits()),
    })
}
/// Fold a unary float op; operand and result are `ty`.
pub fn fold_fun(ty: FloatTy, op: FUnOp, a: Known) -> Option<Known> {
    // The in-svm build (no `libm-floats`) can't fold the libm ops — pass them through unfolded.
    if !cfg!(feature = "libm-floats") && fun_needs_libm(op) {
        return None;
    }
    Some(match ty {
        FloatTy::F32 => Known::F32(fun32(op, a.as_f32()?).to_bits()),
        FloatTy::F64 => Known::F64(fun64(op, a.as_f64()?).to_bits()),
    })
}
/// Fold a float compare (result is `i32` 0/1).
pub fn fold_fcmp(ty: FloatTy, op: FCmpOp, a: Known, b: Known) -> Option<Known> {
    let r = match ty {
        FloatTy::F32 => {
            let (a, b) = (a.as_f32()?, b.as_f32()?);
            match op {
                FCmpOp::Eq => a == b,
                FCmpOp::Ne => a != b,
                FCmpOp::Lt => a < b,
                FCmpOp::Le => a <= b,
                FCmpOp::Gt => a > b,
                FCmpOp::Ge => a >= b,
            }
        }
        FloatTy::F64 => {
            let (a, b) = (a.as_f64()?, b.as_f64()?);
            match op {
                FCmpOp::Eq => a == b,
                FCmpOp::Ne => a != b,
                FCmpOp::Lt => a < b,
                FCmpOp::Le => a <= b,
                FCmpOp::Gt => a > b,
                FCmpOp::Ge => a >= b,
            }
        }
    };
    Some(Known::I32(r as i32))
}
/// Fold a fused multiply-add `a·b + c` (single rounding), matching the interpreter's `mul_add`.
/// FMA needs `libm` (correctly-rounded software FMA → x86 inline-asm + i128), so it folds only with
/// the `libm-floats` feature; the in-svm build leaves `fma` unfolded (sound — it runs at runtime).
#[cfg(feature = "libm-floats")]
pub fn fold_fma(ty: FloatTy, a: Known, b: Known, c: Known) -> Option<Known> {
    Some(match ty {
        // `libm::fmaf`/`fma` is the correctly-rounded IEEE FMA, bit-identical to the interpreter's
        // `mul_add` (a hardware/correctly-rounded FMA).
        FloatTy::F32 => Known::F32(libm::fmaf(a.as_f32()?, b.as_f32()?, c.as_f32()?).to_bits()),
        FloatTy::F64 => Known::F64(libm::fma(a.as_f64()?, b.as_f64()?, c.as_f64()?).to_bits()),
    })
}
#[cfg(not(feature = "libm-floats"))]
pub fn fold_fma(_ty: FloatTy, _a: Known, _b: Known, _c: Known) -> Option<Known> {
    None // in-svm build: FMA needs libm (i128/inline-asm), so leave it unfolded
}
/// The `f64` value of a float operand, promoting `f32` exactly (matching the interpreter).
fn ftoi_input(op: FToI, a: Known) -> Option<f64> {
    Some(match op.parts().0 {
        FloatTy::F32 => a.as_f32()? as f64,
        FloatTy::F64 => a.as_f64()?,
    })
}
/// Fold a **saturating** float→int conversion (`trunc_sat`): NaN → 0, out-of-range saturates —
/// Rust's `as` cast matches wasm exactly, so it never fails.
pub fn fold_ftoi_sat(op: FToI, a: Known) -> Option<Known> {
    let f = ftoi_input(op, a)?;
    let (_, to, signed) = op.parts();
    Some(match (to, signed) {
        (IntTy::I32, true) => Known::I32(f as i32),
        (IntTy::I32, false) => Known::I32(f as u32 as i32),
        (IntTy::I64, true) => Known::I64(f as i64),
        (IntTy::I64, false) => Known::I64(f as u64 as i64),
    })
}
/// Fold a **trapping** float→int conversion (`trunc`) — but only when it would *not* trap (the
/// input is finite and truncates into range). On a NaN/out-of-range input, return `None` so the
/// op is kept and traps at runtime exactly as the source would. Bounds mirror `trunc_trap`.
pub fn fold_ftoi_trap(op: FToI, a: Known) -> Option<Known> {
    let f = ftoi_input(op, a)?;
    let (_, to, signed) = op.parts();
    if f.is_nan() {
        return None;
    }
    #[allow(clippy::manual_range_contains)]
    let in_range = match (to, signed) {
        (IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (IntTy::I64, true) => f >= -9_223_372_036_854_775_808.0 && f < 9_223_372_036_854_775_808.0,
        (IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return None;
    }
    fold_ftoi_sat(op, a)
}
/// Fold an int→float conversion, matching the interpreter's `i_to_f`.
pub fn fold_itof(op: IToF, a: Known) -> Option<Known> {
    Some(match op {
        IToF::I32F32S => Known::F32((a.as_i32()? as f32).to_bits()),
        IToF::I32F32U => Known::F32((a.as_i32()? as u32 as f32).to_bits()),
        IToF::I64F32S => Known::F32((a.as_i64()? as f32).to_bits()),
        IToF::I64F32U => Known::F32((a.as_i64()? as u64 as f32).to_bits()),
        IToF::I32F64S => Known::F64((a.as_i32()? as f64).to_bits()),
        IToF::I32F64U => Known::F64((a.as_i32()? as u32 as f64).to_bits()),
        IToF::I64F64S => Known::F64((a.as_i64()? as f64).to_bits()),
        IToF::I64F64U => Known::F64((a.as_i64()? as u64 as f64).to_bits()),
    })
}
/// Fold a `demote`/`promote`/`reinterpret` cast, matching the interpreter's `cast`.
pub fn fold_cast(op: CastOp, a: Known) -> Option<Known> {
    Some(match op {
        CastOp::Demote => Known::F32((a.as_f64()? as f32).to_bits()),
        CastOp::Promote => Known::F64((a.as_f32()? as f64).to_bits()),
        CastOp::ReinterpI32F32 => Known::F32(a.as_i32()? as u32),
        CastOp::ReinterpF32I32 => Known::I32(a.as_f32()?.to_bits() as i32),
        CastOp::ReinterpI64F64 => Known::F64(a.as_i64()? as u64),
        CastOp::ReinterpF64I64 => Known::I64(a.as_f64()?.to_bits() as i64),
    })
}

// ----- v128 (SIMD) constant folding (mirrors `svm-interp`'s `simd_*` lane helpers exactly) -----
//
// All ops work on raw `[u8; 16]` bytes; float lanes reuse the scalar `fbin*`/`fun*`/`fcmp*` helpers
// above, so the deliberate NaN/rounding fidelity carries over to vectors for free. The common ops
// fold; the exotic ones (saturating add/sub, widen/narrow, int↔float convert, dot, pairwise,
// pmin/pmax, avgr, popcnt, any/all-true, bitmask, q15) pass through to the residual unfolded.

/// Read lane `lane` (`bytes` wide) of a `v128` as a zero-extended `u64`.
fn lane_read(v: &[u8; 16], lane: usize, bytes: usize) -> u64 {
    let mut x = 0u64;
    for k in 0..bytes {
        x |= (v[lane * bytes + k] as u64) << (8 * k);
    }
    x
}

/// Write the low `bytes` of `x` into lane `lane`.
fn lane_write(v: &mut [u8; 16], lane: usize, bytes: usize, x: u64) {
    for k in 0..bytes {
        v[lane * bytes + k] = (x >> (8 * k)) as u8;
    }
}

/// Sign-extend the low `bytes` of a zero-extended lane value to a full `i64`.
fn lane_sext(x: u64, bytes: usize) -> i64 {
    let bits = bytes * 8;
    if bits >= 64 {
        x as i64
    } else {
        let shift = 64 - bits;
        ((x << shift) as i64) >> shift
    }
}

/// Fold a pure `v128` lane op whose operands are all known. `get(i)` returns operand `i`'s constant
/// (or `None`). Returns `None` for a non-foldable / dynamic / not-yet-supported op (which then
/// passes through to the residual unfolded).
pub fn fold_simd(inst: &Inst, get: impl Fn(ValIdx) -> Option<Known>) -> Option<Known> {
    let v = |i: ValIdx| get(i)?.as_v128();
    Some(match *inst {
        Inst::ConstV128(b) => Known::V128(b),
        Inst::Splat { shape, a } => Known::V128(simd_splat(shape, get(a)?.lane_bits())),
        Inst::ExtractLane {
            shape,
            lane,
            signed,
            a,
        } => simd_extract(shape, lane, signed, v(a)?),
        Inst::ReplaceLane { shape, lane, a, b } => {
            Known::V128(simd_replace(shape, lane, v(a)?, get(b)?.lane_bits()))
        }
        Inst::VIntBin { shape, op, a, b } => Known::V128(simd_vint_bin(shape, op, v(a)?, v(b)?)),
        Inst::VIntCmp { shape, op, a, b } => Known::V128(simd_vint_cmp(shape, op, v(a)?, v(b)?)),
        Inst::VIntUn { shape, op, a } => Known::V128(simd_vint_un(shape, op, v(a)?)),
        Inst::VShift { shape, op, a, amt } => {
            Known::V128(simd_vshift(shape, op, v(a)?, get(amt)?.as_i32()? as u32))
        }
        Inst::VFloatBin { shape, op, a, b } => {
            Known::V128(simd_vfloat_bin(shape, op, v(a)?, v(b)?))
        }
        // SIMD `sqrt`/`ceil`/`floor`/`trunc`/nearest per lane need libm (like the scalar path); fold
        // only with `libm-floats`. `abs`/`neg` lanes are pure `core`, so they still fold.
        Inst::VFloatUn { shape, op, a }
            if cfg!(feature = "libm-floats") || !fun_needs_libm(vf_un(op)) =>
        {
            Known::V128(simd_vfloat_un(shape, op, v(a)?))
        }
        Inst::VFloatCmp { shape, op, a, b } => {
            Known::V128(simd_vfloat_cmp(shape, op, v(a)?, v(b)?))
        }
        // SIMD FMA needs libm (i128/inline-asm); fold only with `libm-floats` (else this arm is
        // removed and `VFma` falls through to `_ => return None`, i.e. passes through unfolded).
        #[cfg(feature = "libm-floats")]
        Inst::VFma {
            shape,
            neg,
            a,
            b,
            c,
        } => Known::V128(simd_fma(shape, neg, v(a)?, v(b)?, v(c)?)),
        Inst::VBitBin { op, a, b } => Known::V128(simd_vbit_bin(op, v(a)?, v(b)?)),
        Inst::VNot { a } => Known::V128(simd_vnot(v(a)?)),
        Inst::Bitselect { a, b, mask } => Known::V128(simd_bitselect(v(a)?, v(b)?, v(mask)?)),
        Inst::Shuffle { lanes, a, b } => Known::V128(simd_shuffle(&lanes, v(a)?, v(b)?)),
        Inst::Swizzle { a, b } => Known::V128(simd_swizzle(v(a)?, v(b)?)),
        // Exotic lane ops — each mirrors the matching `svm-interp` `simd_*` helper bit-for-bit (the
        // differential oracle, `tests/specialize.rs::folds_v128_exotic_lane_ops`, cross-checks them).
        Inst::VSatBin { shape, op, a, b } => Known::V128(simd_vsat_bin(shape, op, v(a)?, v(b)?)),
        Inst::VWiden { shape, op, a } => Known::V128(simd_widen(shape, op, v(a)?)),
        Inst::VNarrow { shape, op, a, b } => Known::V128(simd_narrow(shape, op, v(a)?, v(b)?)),
        Inst::VConvert { op, a } => Known::V128(simd_convert(op, v(a)?)),
        Inst::VPMinMax { shape, op, a, b } => Known::V128(simd_pminmax(shape, op, v(a)?, v(b)?)),
        Inst::VPopcnt { a } => {
            let x = v(a)?;
            let mut o = [0u8; 16];
            for i in 0..16 {
                o[i] = x[i].count_ones() as u8;
            }
            Known::V128(o)
        }
        Inst::VAvgr { shape, a, b } => Known::V128(simd_avgr(shape, v(a)?, v(b)?)),
        Inst::VDot { a, b } => Known::V128(simd_dot(v(a)?, v(b)?)),
        Inst::VDotI8 { a, b } => Known::V128(simd_dot_i8(v(a)?, v(b)?)),
        Inst::VExtMul { shape, op, a, b } => Known::V128(simd_extmul(shape, op, v(a)?, v(b)?)),
        Inst::VExtAddPairwise { shape, signed, a } => {
            Known::V128(simd_extadd_pairwise(shape, signed, v(a)?))
        }
        Inst::VQ15MulrSat { a, b } => Known::V128(simd_q15mulr(v(a)?, v(b)?)),
        Inst::VAnyTrue { a } => Known::I32(v(a)?.iter().any(|&b| b != 0) as i32),
        Inst::VAllTrue { shape, a } => Known::I32(simd_all_true(shape, v(a)?)),
        Inst::VBitmask { shape, a } => Known::I32(simd_bitmask(shape, v(a)?)),
        _ => return None,
    })
}

// ---- exotic lane-op helpers (ported from `svm-interp`'s `simd_*`, kept bit-identical) ----------

fn simd_vsat_bin(shape: VShape, op: VSatBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let bits = bytes as u32 * 8;
    // Saturating add/sub exist only for 8/16-bit lanes (wasm), so every value and sum fits `i64` —
    // no need for `i128` (which the svm-IR on-ramp can't translate; see DESIGN.md §20c).
    let max_u = (1i64 << bits) - 1;
    let max_s = (1i64 << (bits - 1)) - 1;
    let min_s = -(1i64 << (bits - 1));
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let (xu, yu) = (
            lane_read(&a, i, bytes) as i64,
            lane_read(&b, i, bytes) as i64,
        );
        let (xs, ys) = (
            lane_sext(lane_read(&a, i, bytes), bytes),
            lane_sext(lane_read(&b, i, bytes), bytes),
        );
        let r = match op {
            VSatBinOp::AddU => (xu + yu).min(max_u),
            VSatBinOp::SubU => (xu - yu).max(0),
            VSatBinOp::AddS => (xs + ys).clamp(min_s, max_s),
            VSatBinOp::SubS => (xs - ys).clamp(min_s, max_s),
        };
        lane_write(&mut o, i, bytes, r as u64);
    }
    o
}

fn simd_widen(out: VShape, op: VWidenOp, a: [u8; 16]) -> [u8; 16] {
    let (low, signed) = op.parts();
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize;
    let base = if low { 0 } else { n };
    let mut o = [0u8; 16];
    for i in 0..n {
        let s = lane_read(&a, base + i, src_bytes);
        let v = if signed {
            lane_sext(s, src_bytes) as u64
        } else {
            s
        };
        lane_write(&mut o, i, out_bytes, v);
    }
    o
}

fn simd_narrow(out: VShape, op: VNarrowOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let out_bytes = out.lane_bytes() as usize;
    let src = out.wider().expect("verifier ensures a wider source");
    let src_bytes = src.lane_bytes() as usize;
    let src_lanes = src.lanes() as usize; // = out.lanes() / 2
    let bits = out_bytes as u32 * 8;
    // Narrow targets 8/16-bit lanes from 16/32-bit sources, so the clamp bounds and sign-extended
    // sources all fit `i64` (no `i128` — the on-ramp can't translate it).
    let (min, max) = match op {
        VNarrowOp::S => (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
        VNarrowOp::U => (0i64, (1i64 << bits) - 1),
    };
    let mut o = [0u8; 16];
    for i in 0..src_lanes {
        let s = lane_sext(lane_read(&a, i, src_bytes), src_bytes);
        lane_write(&mut o, i, out_bytes, s.clamp(min, max) as u64);
    }
    for i in 0..src_lanes {
        let s = lane_sext(lane_read(&b, i, src_bytes), src_bytes);
        lane_write(&mut o, src_lanes + i, out_bytes, s.clamp(min, max) as u64);
    }
    o
}

fn simd_extmul(out: VShape, op: VWidenOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let (low, signed) = op.parts();
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize;
    let base = if low { 0 } else { n };
    let mut o = [0u8; 16];
    for i in 0..n {
        let ra = lane_read(&a, base + i, src_bytes);
        let rb = lane_read(&b, base + i, src_bytes);
        // Widening multiply: source lanes are ≤32-bit, so the 2× product fits 64 bits — `i64` for
        // signed, `u64` for unsigned (`u32×u32` can exceed `i64::MAX` but fits `u64`). `i128` was
        // unnecessary, and the on-ramp can't translate it (DESIGN.md §20c).
        let prod = if signed {
            lane_sext(ra, src_bytes).wrapping_mul(lane_sext(rb, src_bytes)) as u64
        } else {
            ra.wrapping_mul(rb)
        };
        lane_write(&mut o, i, out_bytes, prod);
    }
    o
}

fn simd_extadd_pairwise(out: VShape, signed: bool, a: [u8; 16]) -> [u8; 16] {
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize;
    // Pairwise widen-add of 8/16-bit lanes into 16/32-bit lanes — sums of two ≤16-bit values fit
    // `i64` comfortably (no `i128`, which the on-ramp can't translate).
    let widen = |raw: u64| -> i64 {
        if signed {
            lane_sext(raw, src_bytes)
        } else {
            raw as i64
        }
    };
    let mut o = [0u8; 16];
    for i in 0..n {
        let lo = widen(lane_read(&a, 2 * i, src_bytes));
        let hi = widen(lane_read(&a, 2 * i + 1, src_bytes));
        lane_write(&mut o, i, out_bytes, (lo + hi) as u64);
    }
    o
}

fn simd_q15mulr(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..8 {
        let x = lane_sext(lane_read(&a, i, 2), 2);
        let y = lane_sext(lane_read(&b, i, 2), 2);
        let r = (x * y + 0x4000) >> 15;
        let sat = r.clamp(i16::MIN as i64, i16::MAX as i64);
        lane_write(&mut o, i, 2, sat as u16 as u64);
    }
    o
}

fn simd_avgr(shape: VShape, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let y = lane_read(&b, i, bytes);
        lane_write(&mut o, i, bytes, (x + y + 1) >> 1);
    }
    o
}

fn simd_dot(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..4 {
        let a0 = lane_sext(lane_read(&a, 2 * i, 2), 2) as i32;
        let a1 = lane_sext(lane_read(&a, 2 * i + 1, 2), 2) as i32;
        let b0 = lane_sext(lane_read(&b, 2 * i, 2), 2) as i32;
        let b1 = lane_sext(lane_read(&b, 2 * i + 1, 2), 2) as i32;
        let r = a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1));
        lane_write(&mut o, i, 4, r as u32 as u64);
    }
    o
}

fn simd_dot_i8(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for j in 0..8 {
        let a0 = lane_sext(lane_read(&a, 2 * j, 1), 1) as i32;
        let a1 = lane_sext(lane_read(&a, 2 * j + 1, 1), 1) as i32;
        let b0 = lane_sext(lane_read(&b, 2 * j, 1), 1) as i32;
        let b1 = lane_sext(lane_read(&b, 2 * j + 1, 1), 1) as i32;
        let r = a0 * b0 + a1 * b1; // exact in i32; wraps when written at i16 width
        lane_write(&mut o, j, 2, r as u16 as u64);
    }
    o
}

fn simd_all_true(shape: VShape, a: [u8; 16]) -> i32 {
    let bytes = shape.lane_bytes() as usize;
    (0..shape.lanes() as usize).all(|i| lane_read(&a, i, bytes) != 0) as i32
}

fn simd_bitmask(shape: VShape, a: [u8; 16]) -> i32 {
    let bytes = shape.lane_bytes() as usize;
    let top = bytes as u32 * 8 - 1;
    let mut m = 0i32;
    for i in 0..shape.lanes() as usize {
        m |= (((lane_read(&a, i, bytes) >> top) & 1) as i32) << i;
    }
    m
}

fn simd_pminmax(shape: VShape, op: VPMinMaxOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    // wasm pmin/pmax: a one-sided compare-and-select — pmin(a,b)=b<a?b:a, pmax(a,b)=a<b?b:a — which
    // propagates NaN from the second operand and returns the chosen operand's ±0 (no canonicalization).
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                let r = match op {
                    VPMinMaxOp::Pmin => {
                        if y < x {
                            y
                        } else {
                            x
                        }
                    }
                    VPMinMaxOp::Pmax => {
                        if x < y {
                            y
                        } else {
                            x
                        }
                    }
                };
                lane_write(&mut o, i, 4, r.to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let r = match op {
                    VPMinMaxOp::Pmin => {
                        if y < x {
                            y
                        } else {
                            x
                        }
                    }
                    VPMinMaxOp::Pmax => {
                        if x < y {
                            y
                        } else {
                            x
                        }
                    }
                };
                lane_write(&mut o, i, 8, r.to_bits());
            }
        }
        _ => {} // verifier rejects an integer shape here
    }
    o
}

fn simd_convert(op: VCvtOp, a: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match op {
        VCvtOp::F32x4ConvertI32x4S => {
            for i in 0..4 {
                let x = lane_read(&a, i, 4) as u32 as i32;
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
        }
        VCvtOp::F32x4ConvertI32x4U => {
            for i in 0..4 {
                let x = lane_read(&a, i, 4) as u32;
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
        }
        VCvtOp::I32x4TruncSatF32x4S => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, (x as i32) as u32 as u64);
            }
        }
        VCvtOp::I32x4TruncSatF32x4U => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, (x as u32) as u64);
            }
        }
        VCvtOp::F32x4DemoteF64x2Zero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
            // lanes 2/3 stay zero.
        }
        VCvtOp::F64x2PromoteLowF32x4 => {
            for i in 0..2 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::F64x2ConvertLowI32x4S => {
            for i in 0..2 {
                let x = lane_read(&a, i, 4) as u32 as i32;
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::F64x2ConvertLowI32x4U => {
            for i in 0..2 {
                let x = lane_read(&a, i, 4) as u32;
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::I32x4TruncSatF64x2SZero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as i32) as u32 as u64);
            }
            // lanes 2/3 stay zero.
        }
        VCvtOp::I32x4TruncSatF64x2UZero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as u32) as u64);
            }
            // lanes 2/3 stay zero.
        }
    }
    o
}

fn simd_splat(shape: VShape, bits: u64) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        lane_write(&mut o, i, bytes, bits);
    }
    o
}

fn simd_extract(shape: VShape, lane: u8, signed: bool, v: [u8; 16]) -> Known {
    let bytes = shape.lane_bytes() as usize;
    let lane = (lane as usize).min(shape.lanes() as usize - 1);
    let raw = lane_read(&v, lane, bytes);
    match shape {
        VShape::I8x16 | VShape::I16x8 => {
            let bits = (bytes * 8) as u32;
            let ext = if signed {
                let shift = 32 - bits;
                (((raw as u32) << shift) as i32) >> shift
            } else {
                raw as i32
            };
            Known::I32(ext)
        }
        VShape::I32x4 => Known::I32(raw as i32),
        VShape::I64x2 => Known::I64(raw as i64),
        VShape::F32x4 => Known::F32(raw as u32),
        VShape::F64x2 => Known::F64(raw),
    }
}

fn simd_replace(shape: VShape, lane: u8, mut v: [u8; 16], bits: u64) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let lane = (lane as usize).min(shape.lanes() as usize - 1);
    lane_write(&mut v, lane, bytes, bits);
    v
}

fn simd_vint_bin(shape: VShape, op: VIntBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let y = lane_read(&b, i, bytes);
        let r = match op {
            VIntBinOp::Add => x.wrapping_add(y),
            VIntBinOp::Sub => x.wrapping_sub(y),
            VIntBinOp::Mul => x.wrapping_mul(y),
            VIntBinOp::MinU => x.min(y),
            VIntBinOp::MaxU => x.max(y),
            VIntBinOp::MinS => lane_sext(x, bytes).min(lane_sext(y, bytes)) as u64,
            VIntBinOp::MaxS => lane_sext(x, bytes).max(lane_sext(y, bytes)) as u64,
        };
        lane_write(&mut o, i, bytes, r);
    }
    o
}

fn simd_vint_cmp(shape: VShape, op: VICmpOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let (xu, yu) = (lane_read(&a, i, bytes), lane_read(&b, i, bytes));
        let (xs, ys) = (lane_sext(xu, bytes), lane_sext(yu, bytes));
        let t = match op {
            VICmpOp::Eq => xu == yu,
            VICmpOp::Ne => xu != yu,
            VICmpOp::LtS => xs < ys,
            VICmpOp::LtU => xu < yu,
            VICmpOp::GtS => xs > ys,
            VICmpOp::GtU => xu > yu,
            VICmpOp::LeS => xs <= ys,
            VICmpOp::LeU => xu <= yu,
            VICmpOp::GeS => xs >= ys,
            VICmpOp::GeU => xu >= yu,
        };
        lane_write(&mut o, i, bytes, if t { u64::MAX } else { 0 });
    }
    o
}

fn simd_vint_un(shape: VShape, op: VIntUnOp, a: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_sext(lane_read(&a, i, bytes), bytes);
        let r = match op {
            VIntUnOp::Abs => x.wrapping_abs(),
            VIntUnOp::Neg => x.wrapping_neg(),
        };
        lane_write(&mut o, i, bytes, r as u64);
    }
    o
}

fn simd_vshift(shape: VShape, op: VShiftOp, a: [u8; 16], amt: u32) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let sh = amt & (bytes as u32 * 8 - 1);
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let r = match op {
            VShiftOp::Shl => x << sh,
            VShiftOp::ShrU => x >> sh,
            VShiftOp::ShrS => (lane_sext(x, bytes) >> sh) as u64,
        };
        lane_write(&mut o, i, bytes, r);
    }
    o
}

/// Map a vector float op onto the scalar [`FBinOp`]/[`FUnOp`] so lanes match scalars exactly.
fn vf_bin(op: VFloatBinOp) -> FBinOp {
    match op {
        VFloatBinOp::Add => FBinOp::Add,
        VFloatBinOp::Sub => FBinOp::Sub,
        VFloatBinOp::Mul => FBinOp::Mul,
        VFloatBinOp::Div => FBinOp::Div,
        VFloatBinOp::Min => FBinOp::Min,
        VFloatBinOp::Max => FBinOp::Max,
    }
}
fn vf_un(op: VFloatUnOp) -> FUnOp {
    match op {
        VFloatUnOp::Abs => FUnOp::Abs,
        VFloatUnOp::Neg => FUnOp::Neg,
        VFloatUnOp::Sqrt => FUnOp::Sqrt,
        VFloatUnOp::Ceil => FUnOp::Ceil,
        VFloatUnOp::Floor => FUnOp::Floor,
        VFloatUnOp::Trunc => FUnOp::Trunc,
        VFloatUnOp::Nearest => FUnOp::Nearest,
    }
}

fn simd_vfloat_bin(shape: VShape, op: VFloatBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                lane_write(&mut o, i, 4, fbin32(vf_bin(op), x, y).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                lane_write(&mut o, i, 8, fbin64(vf_bin(op), x, y).to_bits());
            }
        }
        _ => {}
    }
    o
}

fn simd_vfloat_un(shape: VShape, op: VFloatUnOp, a: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, fun32(vf_un(op), x).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 8, fun64(vf_un(op), x).to_bits());
            }
        }
        _ => {}
    }
    o
}

fn simd_vfloat_cmp(shape: VShape, op: VFCmpOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let cmp32 = |x: f32, y: f32| match op {
        VFCmpOp::Eq => x == y,
        VFCmpOp::Ne => x != y,
        VFCmpOp::Lt => x < y,
        VFCmpOp::Gt => x > y,
        VFCmpOp::Le => x <= y,
        VFCmpOp::Ge => x >= y,
    };
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                lane_write(&mut o, i, 4, if cmp32(x, y) { u64::MAX } else { 0 });
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let t = match op {
                    VFCmpOp::Eq => x == y,
                    VFCmpOp::Ne => x != y,
                    VFCmpOp::Lt => x < y,
                    VFCmpOp::Gt => x > y,
                    VFCmpOp::Le => x <= y,
                    VFCmpOp::Ge => x >= y,
                };
                lane_write(&mut o, i, 8, if t { u64::MAX } else { 0 });
            }
        }
        _ => {}
    }
    o
}

// Only compiled with `libm-floats` (its sole caller, the `VFma` fold arm, is `#[cfg]`-gated the same
// way); the in-svm build leaves SIMD FMA unfolded.
#[cfg(feature = "libm-floats")]
fn simd_fma(shape: VShape, neg: bool, a: [u8; 16], b: [u8; 16], c: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                let z = f32::from_bits(lane_read(&c, i, 4) as u32);
                let x = if neg { -x } else { x };
                lane_write(&mut o, i, 4, libm::fmaf(x, y, z).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let z = f64::from_bits(lane_read(&c, i, 8));
                let x = if neg { -x } else { x };
                lane_write(&mut o, i, 8, libm::fma(x, y, z).to_bits());
            }
        }
        _ => {}
    }
    o
}

fn simd_vbit_bin(op: VBitBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = match op {
            VBitBinOp::And => a[i] & b[i],
            VBitBinOp::Or => a[i] | b[i],
            VBitBinOp::Xor => a[i] ^ b[i],
            VBitBinOp::AndNot => a[i] & !b[i],
        };
    }
    o
}

fn simd_vnot(a: [u8; 16]) -> [u8; 16] {
    a.map(|x| !x)
}

fn simd_bitselect(a: [u8; 16], b: [u8; 16], mask: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = (a[i] & mask[i]) | (b[i] & !mask[i]);
    }
    o
}

fn simd_shuffle(lanes: &[u8; 16], a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        let sel = lanes[i] as usize;
        o[i] = if sel < 16 {
            a[sel]
        } else if sel < 32 {
            b[sel - 16]
        } else {
            0
        };
    }
    o
}

fn simd_swizzle(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        let sel = b[i] as usize;
        o[i] = if sel < 16 { a[sel] } else { 0 };
    }
    o
}

/// Constant-fold an integer binary op, mirroring the interpreter's `bin32`/`bin64` exactly
/// (wrapping arithmetic; shifts/rotates mod bitwidth). Returns `None` for the trapping cases
/// so the op is preserved and traps at runtime as the source would.
pub fn fold_int_bin(ty: IntTy, op: BinOp, a: Known, b: Known) -> Option<Known> {
    match ty {
        IntTy::I32 => {
            let (a, b) = (a.as_i32()?, b.as_i32()?);
            let r = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::DivS => {
                    if b == 0 || (a == i32::MIN && b == -1) {
                        return None;
                    }
                    a.wrapping_div(b)
                }
                BinOp::DivU => {
                    if b == 0 {
                        return None;
                    }
                    ((a as u32) / (b as u32)) as i32
                }
                BinOp::RemS => {
                    if b == 0 {
                        return None;
                    }
                    a.wrapping_rem(b)
                }
                BinOp::RemU => {
                    if b == 0 {
                        return None;
                    }
                    ((a as u32) % (b as u32)) as i32
                }
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::ShrS => a.wrapping_shr(b as u32),
                BinOp::ShrU => ((a as u32).wrapping_shr(b as u32)) as i32,
                BinOp::Rotl => a.rotate_left(b as u32),
                BinOp::Rotr => a.rotate_right(b as u32),
            };
            Some(Known::I32(r))
        }
        IntTy::I64 => {
            let (a, b) = (a.as_i64()?, b.as_i64()?);
            let r = match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::DivS => {
                    if b == 0 || (a == i64::MIN && b == -1) {
                        return None;
                    }
                    a.wrapping_div(b)
                }
                BinOp::DivU => {
                    if b == 0 {
                        return None;
                    }
                    ((a as u64) / (b as u64)) as i64
                }
                BinOp::RemS => {
                    if b == 0 {
                        return None;
                    }
                    a.wrapping_rem(b)
                }
                BinOp::RemU => {
                    if b == 0 {
                        return None;
                    }
                    ((a as u64) % (b as u64)) as i64
                }
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::Shl => a.wrapping_shl(b as u32),
                BinOp::ShrS => a.wrapping_shr(b as u32),
                BinOp::ShrU => ((a as u64).wrapping_shr(b as u32)) as i64,
                BinOp::Rotl => a.rotate_left(b as u32),
                BinOp::Rotr => a.rotate_right(b as u32),
            };
            Some(Known::I64(r))
        }
    }
}

/// Constant-fold an integer comparison (result is `i32` 0/1), mirroring `cmp32`/`cmp64`.
pub fn fold_int_cmp(ty: IntTy, op: CmpOp, a: Known, b: Known) -> Option<Known> {
    let r = match ty {
        IntTy::I32 => {
            let (a, b) = (a.as_i32()?, b.as_i32()?);
            match op {
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
            }
        }
        IntTy::I64 => {
            let (a, b) = (a.as_i64()?, b.as_i64()?);
            match op {
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
            }
        }
    };
    Some(Known::I32(r as i32))
}

/// Constant-fold a unary integer op, mirroring `intun32`/`intun64`.
pub fn fold_int_un(ty: IntTy, op: IntUnOp, a: Known) -> Option<Known> {
    match ty {
        IntTy::I32 => {
            let a = a.as_i32()?;
            let r = match op {
                IntUnOp::Clz => (a as u32).leading_zeros() as i32,
                IntUnOp::Ctz => (a as u32).trailing_zeros() as i32,
                IntUnOp::Popcnt => (a as u32).count_ones() as i32,
                IntUnOp::Extend8S => (a as i8) as i32,
                IntUnOp::Extend16S => (a as i16) as i32,
                IntUnOp::Extend32S => a,
            };
            Some(Known::I32(r))
        }
        IntTy::I64 => {
            let a = a.as_i64()?;
            let r = match op {
                IntUnOp::Clz => (a as u64).leading_zeros() as i64,
                IntUnOp::Ctz => (a as u64).trailing_zeros() as i64,
                IntUnOp::Popcnt => (a as u64).count_ones() as i64,
                IntUnOp::Extend8S => (a as i8) as i64,
                IntUnOp::Extend16S => (a as i16) as i64,
                IntUnOp::Extend32S => (a as i32) as i64,
            };
            Some(Known::I64(r))
        }
    }
}

/// Whether known value `i` equals the signed constant `v` (width-agnostic for `0`/`1`/`-1`).
fn known_is(known: &[Option<Known>], i: ValIdx, v: i64) -> bool {
    match get(known, i) {
        Some(Known::I32(x)) => x as i64 == v,
        Some(Known::I64(x)) => x == v,
        _ => false,
    }
}

/// Identities that fold to a *constant* even with one operand unknown (absorbing elements and
/// self-cancellation): `x*0`/`x&0` → 0, `x|-1` → -1, `x-x`/`x^x` → 0, `x%1` → 0. Sound for any `x`
/// (none of these traps on these operands), so they need only one known operand — or `a == b`.
fn fold_absorbing(
    ty: IntTy,
    op: BinOp,
    a: ValIdx,
    b: ValIdx,
    known: &[Option<Known>],
) -> Option<Known> {
    let zero = match ty {
        IntTy::I32 => Known::I32(0),
        IntTy::I64 => Known::I64(0),
    };
    let is = |i, v| known_is(known, i, v);
    match op {
        BinOp::Mul | BinOp::And if is(a, 0) || is(b, 0) => Some(zero),
        BinOp::Or if is(a, -1) || is(b, -1) => Some(match ty {
            IntTy::I32 => Known::I32(-1),
            IntTy::I64 => Known::I64(-1),
        }),
        BinOp::Sub | BinOp::Xor if a == b => Some(zero),
        BinOp::RemS | BinOp::RemU if is(b, 1) => Some(zero),
        _ => None,
    }
}

/// If a single-result instruction is a *copy* of an earlier value — a constant-condition `select`,
/// or an algebraic identity (`x+0`/`x-0`/`x*1`/`x<<0`/`x/1`, `x|0`/`x&-1`/`x^0`, `x&x`/`x|x`) —
/// return the source value its result should forward to. (Identities that fold to a *constant* go
/// through [`fold_absorbing`] instead.)
fn forward_to_operand(inst: &Inst, known: &[Option<Known>]) -> Option<ValIdx> {
    let is = |i, v| known_is(known, i, v);
    match *inst {
        Inst::Select { cond, a, b } => {
            if a == b {
                return Some(a); // equal arms: the result is that value whatever the condition
            }
            let c = get(known, cond)?.as_i32()?;
            Some(if c != 0 { a } else { b })
        }
        Inst::IntBin { op, a, b, .. } => match op {
            BinOp::Add if is(a, 0) => Some(b),
            BinOp::Add if is(b, 0) => Some(a),
            BinOp::Sub if is(b, 0) => Some(a),
            BinOp::Mul if is(a, 1) => Some(b),
            BinOp::Mul if is(b, 1) => Some(a),
            BinOp::Or if is(a, 0) => Some(b),
            BinOp::Or if is(b, 0) || a == b => Some(a),
            BinOp::And if is(a, -1) => Some(b),
            BinOp::And if is(b, -1) || a == b => Some(a),
            BinOp::Xor if is(a, 0) => Some(b),
            BinOp::Xor if is(b, 0) => Some(a),
            BinOp::Shl | BinOp::ShrS | BinOp::ShrU | BinOp::Rotl | BinOp::Rotr if is(b, 0) => {
                Some(a)
            }
            // `x / 1` is deliberately *not* forwarded: division is not removable-if-dead (DCE keeps
            // it conservatively, as a possible trap), so forwarding would leave a dead `div` behind
            // rather than shrinking. `x % 1 → 0` is handled by `fold_absorbing` (an in-place const).
            _ => None,
        },
        _ => None,
    }
}

/// Intra-block copy propagation. Rewrites every use of a value that is a copy of an earlier value
/// (see [`forward_to_operand`]) to that earlier value; the now-unused copy instruction is removed by
/// the following DCE pass. Index-stable (no instruction is removed here). Sound because an operand
/// only references an earlier value in the same block, which dominates the use.
fn copy_propagate(b: &Block, fn_results: &[usize]) -> Block {
    let mut known: Vec<Option<Known>> = vec![None; b.params.len()];
    // `repl[v]` is the value `v` forwards to (its root); params and non-copies map to themselves.
    let mut repl: Vec<ValIdx> = (0..b.params.len() as u32).collect();
    let mut insts = b.insts.clone();
    let mut next = b.params.len() as u32;
    for inst in insts.iter_mut() {
        // Compose with prior forwarding first, so an operand always names its root.
        map_operands(inst, &mut |o| repl[o as usize]);
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            let root = match forward_to_operand(inst, &known) {
                Some(src) => repl[src as usize],
                None => next,
            };
            repl.push(root);
            known.push(const_value(inst));
            next += 1;
        } else {
            for _ in 0..rc {
                repl.push(next);
                known.push(None);
                next += 1;
            }
        }
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |o| repl[o as usize]);
    Block {
        params: b.params.clone(),
        insts,
        term,
    }
}

/// Intra-block common-subexpression elimination. Within a block, a **pure** instruction (no trap, no
/// memory access, no side effect — [`svm_ir::Inst::effects`]) whose operation *and* operands exactly
/// match an earlier pure instruction computes the very same value, so its uses are rewritten to that
/// earlier result and the redundant instruction is left dead for the following DCE pass. Operands are
/// canonicalized to their CSE roots first (exactly as [`copy_propagate`] does), so equal expressions
/// built from equal subexpressions are caught too. Index-stable (nothing is removed here).
///
/// Only pure ops are eligible: purity means the result is a deterministic function of the operands
/// with no trap or effect, so two matching pure ops are interchangeable. A load/atomic/call (memory
/// may change between them, they may trap or have effects) is never a CSE candidate — the effects
/// table draws that line. Restricted to a single block, so the earlier definition trivially dominates
/// the use and no block-parameter threading is needed (that is the job of the later global GVN).
fn local_cse(b: &Block, fn_results: &[usize]) -> Block {
    // `repl[v]` is the value `v` forwards to (its CSE root); params and non-redundant ops map to self.
    let mut repl: Vec<ValIdx> = (0..b.params.len() as u32).collect();
    let mut insts = b.insts.clone();
    let mut next = b.params.len() as u32;
    // Canonicalized pure instructions seen so far, paired with the value each one defines.
    let mut seen: Vec<(Inst, ValIdx)> = Vec::new();
    for inst in insts.iter_mut() {
        // Compose with prior forwarding first, so operands name their roots before we compare.
        map_operands(inst, &mut |o| repl[o as usize]);
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            let root = if inst.effects().is_pure() {
                match seen.iter().find(|(prev, _)| *prev == *inst) {
                    Some((_, e)) => *e, // a matching earlier pure op — reuse its result
                    None => {
                        seen.push((inst.clone(), next));
                        next
                    }
                }
            } else {
                next
            };
            repl.push(root);
            next += 1;
        } else {
            for _ in 0..rc {
                repl.push(next);
                next += 1;
            }
        }
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |o| repl[o as usize]);
    Block {
        params: b.params.clone(),
        insts,
        term,
    }
}

/// Resolve a conditional terminator to an unconditional `br` when its selector is a known
/// constant, using the interpreter's exact selection rule. Non-constant selectors (and the
/// already-unconditional terminators) are returned unchanged.
pub(crate) fn resolve_term(t: &Terminator, known: &[Option<Known>]) -> Terminator {
    match t {
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            // Coincident targets — same block *and* same args — go the same place whatever the
            // condition, so the branch is unconditional (the `cond` computation becomes dead for DCE).
            if then_blk == else_blk && then_args == else_args {
                return Terminator::Br {
                    target: *then_blk,
                    args: then_args.clone(),
                };
            }
            match get(known, *cond).and_then(Known::as_i32) {
                Some(c) if c != 0 => Terminator::Br {
                    target: *then_blk,
                    args: then_args.clone(),
                },
                Some(_) => Terminator::Br {
                    target: *else_blk,
                    args: else_args.clone(),
                },
                None => t.clone(),
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            // Every edge coincides with the default ⇒ unconditional (the `idx` computation dies).
            if targets.iter().all(|e| e == default) {
                return Terminator::Br {
                    target: default.0,
                    args: default.1.clone(),
                };
            }
            match get(known, *idx).and_then(Known::as_i32) {
                Some(c) => {
                    let (target, args) = targets.get(c as u32 as usize).unwrap_or(default);
                    Terminator::Br {
                        target: *target,
                        args: args.clone(),
                    }
                }
                None => t.clone(),
            }
        }
        other => other.clone(),
    }
}

/// The block successors reachable through a terminator (for the reachability walk).
fn term_successors(t: &Terminator) -> Vec<u32> {
    // The canonical successor extraction lives in `cfg` (OPT.md Phase 1b); `prune_unreachable` only
    // walks these to mark reachability, so `cfg::successors`'s deduplication is harmless here.
    cfg::successors(t)
}

/// Rewrite the `BlockIdx` targets of a terminator through an old→new index map. Only the
/// branching terminators carry targets; everything else is left as-is.
fn remap_targets(t: &mut Terminator, map: &[u32]) {
    match t {
        Terminator::Br { target, .. } => *target = map[*target as usize],
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => {
            *then_blk = map[*then_blk as usize];
            *else_blk = map[*else_blk as usize];
        }
        Terminator::BrTable {
            targets, default, ..
        } => {
            for (tg, _) in targets.iter_mut() {
                *tg = map[*tg as usize];
            }
            default.0 = map[default.0 as usize];
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => {}
    }
}

/// Drop blocks unreachable from the entry (block 0) and renumber the survivors, remapping
/// terminator targets. Every successor of a reachable block is itself reachable, so every
/// remapped target has a valid new index.
fn prune_unreachable(blocks: Vec<Block>) -> Vec<Block> {
    let n = blocks.len();
    let mut reachable = vec![false; n];
    let mut stack = vec![0usize];
    if n > 0 {
        reachable[0] = true;
    }
    while let Some(b) = stack.pop() {
        for s in term_successors(&blocks[b].term) {
            let s = s as usize;
            if s < n && !reachable[s] {
                reachable[s] = true;
                stack.push(s);
            }
        }
    }

    // old index → new index (only meaningful for reachable blocks).
    let mut map = vec![0u32; n];
    let mut next = 0u32;
    for (i, &live) in reachable.iter().enumerate() {
        if live {
            map[i] = next;
            next += 1;
        }
    }

    blocks
        .into_iter()
        .enumerate()
        .filter(|(i, _)| reachable[*i])
        .map(|(_, mut b)| {
            remap_targets(&mut b.term, &map);
            b
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// Stage 0.x: intra-block dead-value elimination.
// ---------------------------------------------------------------------------------------

/// Whether a dead instruction (no live results) is safe to **remove**. True only for the
/// ops that are *pure* or a pure *read* and *cannot trap* and have *no side effect*, so deleting one
/// changes nothing observable. Everything else — anything that can fault (loads, atomics, trapping
/// float→int, `cap.self.get`), writes memory or state (stores, `gc.roots`), transfers control /
/// spawns / blocks (calls, `cap`/`cont`/`thread`/`memory.wait` ops, fences) — is **kept**. The safe
/// default falls out of the classification: a spurious effect only forgoes a removal, never changes
/// behavior.
///
/// This delegates to the single source of truth, [`svm_ir::Inst::effects`] (see `OPT.md` Phase 1a),
/// so the optimizer and every future pass share one purity oracle rather than each carrying its own
/// whitelist.
pub fn is_removable_if_dead(inst: &Inst) -> bool {
    inst.effects().removable_if_dead()
}

/// Apply `f` to **every value operand** of an instruction, in place. Exhaustive on purpose
/// (no wildcard arm): adding an `Inst` variant that carries a `ValIdx` must fail to compile
/// here rather than silently skip an operand and miscompile after renumbering. `FuncIdx`
/// immediates (`RefFunc`/`ThreadSpawn::func`) are *not* value operands and are left alone.
pub fn map_operands(inst: &mut Inst, f: &mut impl FnMut(ValIdx) -> ValIdx) {
    match inst {
        // No value operands.
        Inst::ConstI32(_)
        | Inst::ConstI64(_)
        | Inst::ConstF32(_)
        | Inst::ConstF64(_)
        | Inst::ConstV128(_)
        | Inst::RefFunc { .. }
        | Inst::CapSelfCount
        | Inst::CapSelfAttest
        | Inst::VcpuTlsGet
        | Inst::DurableShadowBase
        | Inst::AtomicFence { .. }
        | Inst::SimdWidthBytes => {}

        // Exactly one operand, named `a`.
        Inst::IntUn { a, .. }
        | Inst::Eqz { a, .. }
        | Inst::Convert { a, .. }
        | Inst::FUn { a, .. }
        | Inst::FToISat { a, .. }
        | Inst::FToITrap { a, .. }
        | Inst::IToFConv { a, .. }
        | Inst::Cast { a, .. }
        | Inst::PtrCast { a, .. }
        | Inst::Load { addr: a, .. }
        | Inst::AtomicLoad { addr: a, .. }
        | Inst::V128Load { addr: a, .. }
        | Inst::CapSelfGet { idx: a }
        | Inst::VcpuTlsSet { val: a }
        | Inst::Suspend { value: a }
        | Inst::SetJmp { buf: a }
        | Inst::ThreadJoin { handle: a }
        | Inst::Splat { a, .. }
        | Inst::ExtractLane { a, .. }
        | Inst::VIntUn { a, .. }
        | Inst::VWiden { a, .. }
        | Inst::VConvert { a, .. }
        | Inst::VPopcnt { a, .. }
        | Inst::VExtAddPairwise { a, .. }
        | Inst::VAnyTrue { a, .. }
        | Inst::VAllTrue { a, .. }
        | Inst::VBitmask { a, .. }
        | Inst::VFloatUn { a, .. }
        | Inst::VNot { a, .. } => {
            *a = f(*a);
        }

        // Exactly two operands, named `a` and `b`.
        Inst::IntBin { a, b, .. }
        | Inst::IntCmp { a, b, .. }
        | Inst::FBin { a, b, .. }
        | Inst::FCmp { a, b, .. }
        | Inst::PtrAdd { a, b }
        | Inst::Store {
            addr: a, value: b, ..
        }
        | Inst::AtomicStore {
            addr: a, value: b, ..
        }
        | Inst::V128Store {
            addr: a, value: b, ..
        }
        | Inst::AtomicRmw {
            addr: a, value: b, ..
        }
        | Inst::MemoryNotify {
            addr: a, count: b, ..
        }
        | Inst::ContNew { func: a, sp: b }
        | Inst::ContResume { k: a, arg: b }
        | Inst::CapSelfResolve {
            name_ptr: a,
            name_len: b,
        }
        | Inst::LongJmp { buf: a, val: b }
        | Inst::ThreadSpawn { sp: a, arg: b, .. }
        | Inst::ReplaceLane { a, b, .. }
        | Inst::VIntBin { a, b, .. }
        | Inst::VIntCmp { a, b, .. }
        | Inst::VFloatCmp { a, b, .. }
        | Inst::VShift { a, amt: b, .. }
        | Inst::VSatBin { a, b, .. }
        | Inst::VNarrow { a, b, .. }
        | Inst::VPMinMax { a, b, .. }
        | Inst::VAvgr { a, b, .. }
        | Inst::VDot { a, b }
        | Inst::VDotI8 { a, b }
        | Inst::VExtMul { a, b, .. }
        | Inst::VQ15MulrSat { a, b }
        | Inst::VFloatBin { a, b, .. }
        | Inst::VBitBin { a, b, .. }
        | Inst::Shuffle { a, b, .. }
        | Inst::Swizzle { a, b } => {
            *a = f(*a);
            *b = f(*b);
        }

        // Three operands.
        Inst::Select { cond, a, b } => {
            *cond = f(*cond);
            *a = f(*a);
            *b = f(*b);
        }
        // Bulk-memory ops (D62): dst, src/val, len — all value operands.
        Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
            *dst = f(*dst);
            *src = f(*src);
            *len = f(*len);
        }
        Inst::MemFill { dst, val, len } => {
            *dst = f(*dst);
            *val = f(*val);
            *len = f(*len);
        }
        Inst::Bitselect { a, b, mask } => {
            *a = f(*a);
            *b = f(*b);
            *mask = f(*mask);
        }
        // Scalar / vector fused multiply-add: `a·b + c`.
        Inst::Fma { a, b, c, .. } | Inst::VFma { a, b, c, .. } => {
            *a = f(*a);
            *b = f(*b);
            *c = f(*c);
        }
        Inst::CapSelfLabel {
            handle,
            buf_ptr,
            buf_cap,
        } => {
            *handle = f(*handle);
            *buf_ptr = f(*buf_ptr);
            *buf_cap = f(*buf_cap);
        }
        Inst::AtomicCmpxchg {
            addr,
            expected,
            replacement,
            ..
        } => {
            *addr = f(*addr);
            *expected = f(*expected);
            *replacement = f(*replacement);
        }
        Inst::MemoryWait {
            addr,
            expected,
            timeout,
            ..
        } => {
            *addr = f(*addr);
            *expected = f(*expected);
            *timeout = f(*timeout);
        }
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => {
            *heap_lo = f(*heap_lo);
            *heap_hi = f(*heap_hi);
            *mask = f(*mask);
            *buf = f(*buf);
            *cap = f(*cap);
        }

        // Variable-length operand lists.
        Inst::Call { args, .. } => {
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Inst::CallIndirect { idx, args, .. } => {
            *idx = f(*idx);
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Inst::CapCall { handle, args, .. } | Inst::CallImport { handle, args, .. } => {
            *handle = f(*handle);
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
    }
}

/// Apply `f` to every **value** operand of a terminator (the branch condition / table index,
/// all edge arguments, return / tail-call arguments). Block-index *targets* are untouched —
/// those are remapped separately by [`remap_targets`].
pub fn map_term_operands(t: &mut Terminator, f: &mut impl FnMut(ValIdx) -> ValIdx) {
    match t {
        Terminator::Br { args, .. } => {
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Terminator::BrIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            *cond = f(*cond);
            for v in then_args.iter_mut().chain(else_args.iter_mut()) {
                *v = f(*v);
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            *idx = f(*idx);
            for (_, args) in targets.iter_mut() {
                for v in args.iter_mut() {
                    *v = f(*v);
                }
            }
            for v in default.1.iter_mut() {
                *v = f(*v);
            }
        }
        Terminator::Return(vals) => {
            for v in vals.iter_mut() {
                *v = f(*v);
            }
        }
        Terminator::ReturnCall { args, .. } => {
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Terminator::ReturnCallIndirect { idx, args, .. } => {
            *idx = f(*idx);
            for v in args.iter_mut() {
                *v = f(*v);
            }
        }
        Terminator::Unreachable => {}
    }
}

/// Visit (read-only) every value operand of an instruction. Implemented on a throwaway clone
/// through [`map_operands`], so there is a single source of truth for "what the operands are".
pub(crate) fn each_operand(inst: &Inst, mut visit: impl FnMut(ValIdx)) {
    let mut tmp = inst.clone();
    map_operands(&mut tmp, &mut |v| {
        visit(v);
        v
    });
}

/// Remove dead values from one block: a backward liveness sweep marks every value used by a
/// kept instruction or the terminator, removable instructions whose results are all dead are
/// dropped, and the survivors are renumbered with every operand rewritten through the new map.
fn dce_block(b: &Block, fn_results: &[usize]) -> Block {
    let nparams = b.params.len() as u32;

    // The first result index of each instruction, and the total value count of the block.
    let mut result_start: Vec<u32> = Vec::with_capacity(b.insts.len());
    let mut next = nparams;
    for inst in &b.insts {
        result_start.push(next);
        next += inst.result_count(fn_results) as u32;
    }
    let total = next as usize;

    // Liveness: terminator operands are roots; then sweep instructions back to front, keeping
    // any with a live result (or that is not removable) and propagating liveness to its operands.
    let mut live = vec![false; total];
    {
        let mut term = b.term.clone();
        map_term_operands(&mut term, &mut |v| {
            live[v as usize] = true;
            v
        });
    }
    let mut keep = vec![false; b.insts.len()];
    for i in (0..b.insts.len()).rev() {
        let inst = &b.insts[i];
        let rc = inst.result_count(fn_results) as u32;
        let start = result_start[i];
        let any_live = (0..rc).any(|k| live[(start + k) as usize]);
        if any_live || !is_removable_if_dead(inst) {
            keep[i] = true;
            each_operand(inst, |v| live[v as usize] = true);
        }
    }

    // Old → new value index. Params keep their indices; kept results pack down after them;
    // removed results have no new index (they are provably unused, so never looked up).
    let mut map: Vec<Option<u32>> = vec![None; total];
    for p in 0..nparams {
        map[p as usize] = Some(p);
    }
    let mut new_next = nparams;
    for (i, &start) in result_start.iter().enumerate() {
        let rc = b.insts[i].result_count(fn_results) as u32;
        if keep[i] {
            for k in 0..rc {
                map[(start + k) as usize] = Some(new_next);
                new_next += 1;
            }
        }
    }
    let lookup = |v: ValIdx| map[v as usize].expect("a live operand must have a new index");

    // Emit the survivors with operands (and the terminator) rewritten through the map.
    let mut insts = Vec::with_capacity(new_next as usize);
    for (i, inst) in b.insts.iter().enumerate() {
        if keep[i] {
            let mut inst = inst.clone();
            map_operands(&mut inst, &mut |v| lookup(v));
            insts.push(inst);
        }
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |v| lookup(v));

    Block {
        params: b.params.clone(),
        insts,
        term,
    }
}

// ---------------------------------------------------------------------------------------
// CFG cleanup: block merging + dead block-parameter elimination.
// ---------------------------------------------------------------------------------------

/// Total number of SSA values a block defines: its parameters plus every instruction result.
fn val_count(b: &Block, fn_results: &[usize]) -> u32 {
    let mut n = b.params.len() as u32;
    for inst in &b.insts {
        n += inst.result_count(fn_results) as u32;
    }
    n
}

/// Number of incoming edges to each block (counting multiplicity — a `br_table` listing a block
/// twice counts twice), so a count of exactly 1 means a single, unique predecessor edge.
fn pred_counts(blocks: &[Block]) -> Vec<u32> {
    let mut c = vec![0u32; blocks.len()];
    for b in blocks {
        for s in term_successors(&b.term) {
            c[s as usize] += 1;
        }
    }
    c
}

/// Merge any block reached by exactly one edge — an unconditional `br` from its sole predecessor —
/// into that predecessor, to a fixpoint. The successor's parameters bind to the branch arguments
/// and its body/terminator are appended (operands renumbered). The entry block is never merged
/// away. This collapses the `br`-chains the specializer emits into straight-line code.
fn merge_blocks(mut blocks: Vec<Block>, fn_results: &[usize]) -> Vec<Block> {
    loop {
        let preds = pred_counts(&blocks);
        // Find a predecessor `a` whose terminator is an unconditional `br` to a mergeable `b`.
        let mut found = None;
        for (a, blk) in blocks.iter().enumerate() {
            if let Terminator::Br { target, .. } = blk.term {
                let b = target as usize;
                if b != a && b != 0 && preds[b] == 1 {
                    found = Some((a, b));
                    break;
                }
            }
        }
        let (a, b) = match found {
            Some(pair) => pair,
            None => return blocks,
        };

        // Pull what we need out of both blocks before mutating.
        let args: Vec<ValIdx> = match &blocks[a].term {
            Terminator::Br { args, .. } => args.clone(),
            _ => unreachable!("selected block must end in `br`"),
        };
        let base = val_count(&blocks[a], fn_results);
        let nparams_b = blocks[b].params.len() as u32;
        let b_insts = blocks[b].insts.clone();
        let b_term = blocks[b].term.clone();

        // Remap a B-local value: a parameter becomes the matching branch argument; an instruction
        // result moves to a fresh index appended after A's existing values.
        let remap = |v: ValIdx| -> ValIdx {
            if v < nparams_b {
                args[v as usize]
            } else {
                base + (v - nparams_b)
            }
        };

        let a_blk = &mut blocks[a];
        for mut inst in b_insts {
            map_operands(&mut inst, &mut |v| remap(v));
            a_blk.insts.push(inst);
        }
        let mut term = b_term;
        map_term_operands(&mut term, &mut |v| remap(v));
        a_blk.term = term;

        remove_block(&mut blocks, b);
    }
}

/// Remove block `b` and renumber the surviving blocks' terminator targets. Nothing references `b`
/// at the call site (its sole predecessor has just absorbed it).
fn remove_block(blocks: &mut Vec<Block>, b: usize) {
    let old_len = blocks.len();
    blocks.remove(b);
    let map: Vec<u32> = (0..old_len)
        .map(|i| if i > b { (i - 1) as u32 } else { i as u32 })
        .collect();
    for blk in blocks.iter_mut() {
        remap_targets(&mut blk.term, &map);
    }
}

/// Drop block parameters that are never referenced within their block, and the matching argument
/// in every predecessor edge. One pass over all blocks (cascades are caught by the outer fixpoint).
/// The entry block's parameters are the function signature and are never dropped.
fn drop_dead_params(blocks: Vec<Block>, fn_results: &[usize]) -> Vec<Block> {
    let n = blocks.len();
    // Dead parameter positions per block (entry excluded).
    let mut dropped: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, blk) in blocks.iter().enumerate().skip(1) {
        let used = used_values(blk, fn_results);
        for (p, &u) in used.iter().take(blk.params.len()).enumerate() {
            if !u {
                dropped[b].push(p);
            }
        }
    }
    if dropped.iter().all(Vec::is_empty) {
        return blocks;
    }

    // Renumber each block to remove its own dead params, then drop the matching edge arguments.
    let mut out: Vec<Block> = blocks
        .iter()
        .enumerate()
        .map(|(b, blk)| {
            if dropped[b].is_empty() {
                blk.clone()
            } else {
                remove_params(blk, &dropped[b])
            }
        })
        .collect();
    for blk in out.iter_mut() {
        drop_edge_args(&mut blk.term, &dropped);
    }
    out
}

/// Which SSA values a block references (as an instruction or terminator operand).
fn used_values(b: &Block, fn_results: &[usize]) -> Vec<bool> {
    let mut used = vec![false; val_count(b, fn_results) as usize];
    for inst in &b.insts {
        each_operand(inst, |v| used[v as usize] = true);
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |v| {
        used[v as usize] = true;
        v
    });
    used
}

/// Rebuild a block with the parameters at `dropped` positions removed, renumbering every value
/// (the dropped params are unused, so no operand ever references them).
fn remove_params(b: &Block, dropped: &[usize]) -> Block {
    let nparams = b.params.len();
    let is_dropped = |p: usize| dropped.contains(&p);
    // old value index -> new value index (None only for the dropped params, never referenced).
    let mut map: Vec<Option<u32>> = Vec::new();
    let mut next = 0u32;
    for p in 0..nparams {
        if is_dropped(p) {
            map.push(None);
        } else {
            map.push(Some(next));
            next += 1;
        }
    }
    // Instruction results all shift down by the number of dropped params.
    let drop_n = dropped.len() as u32;
    let lookup = move |v: ValIdx| -> ValIdx {
        if (v as usize) < nparams {
            map[v as usize].expect("a dropped parameter must be unused")
        } else {
            v - drop_n
        }
    };

    let params: Vec<ValType> = b
        .params
        .iter()
        .enumerate()
        .filter(|(p, _)| !is_dropped(*p))
        .map(|(_, t)| *t)
        .collect();
    let mut insts = b.insts.clone();
    for inst in insts.iter_mut() {
        map_operands(inst, &mut |v| lookup(v));
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |v| lookup(v));
    Block {
        params,
        insts,
        term,
    }
}

// ---------------------------------------------------------------------------------------
// Jump threading (OPT.md Phase 2): thread an edge through an empty conditional forwarder.
// ---------------------------------------------------------------------------------------

/// The constants a block defines, indexed by block-local value: parameters are unknown, and each
/// single-result instruction contributes its literal value (after folding, every folded op has
/// become a `const`). This is the same seeding as [`fold_block`], minus the folding — it runs after
/// `fold_block` in the fixpoint, so operands are already materialized where they can be.
fn block_consts(b: &Block, fn_results: &[usize]) -> Vec<Option<Known>> {
    let mut known: Vec<Option<Known>> = vec![None; b.params.len()];
    for inst in &b.insts {
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            known.push(const_value(inst));
        } else {
            for _ in 0..rc {
                known.push(None);
            }
        }
    }
    known
}

/// If out-edge `(target, args)` reaches an **empty conditional forwarder** — a block with no
/// instructions whose terminator is a `br_if`/`br_table` on one of its own parameters — and the
/// value the predecessor passes for that selector parameter is a known constant, resolve the
/// forwarder's branch and return the edge that skips it: the resolved target, with its arguments
/// mapped back through `args` to values valid in the predecessor.
///
/// Sound because the forwarder has **no instructions** (no side effects, no defs): entering it only
/// selects a branch from a parameter, so threading the predecessor straight to the resolved target
/// with the same argument values is observationally identical. Since the forwarder defines nothing,
/// every value its terminator names is a parameter index `j`, which binds to `args[j]` on this edge.
fn thread_edge(
    target: u32,
    args: &[ValIdx],
    blocks: &[Block],
    q_consts: &[Option<Known>],
) -> Option<(u32, Vec<ValIdx>)> {
    let b = &blocks[target as usize];
    if !b.insts.is_empty() {
        return None; // only empty forwarders: nothing to clone into the predecessor
    }
    // Resolve the forwarder's branch using the constant the predecessor passes for its selector.
    let (rt, rargs): (u32, &Vec<ValIdx>) = match &b.term {
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let sel = *cond as usize;
            if sel >= args.len() {
                return None;
            }
            let c = get(q_consts, args[sel]).and_then(Known::as_i32)?;
            if c != 0 {
                (*then_blk, then_args)
            } else {
                (*else_blk, else_args)
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let sel = *idx as usize;
            if sel >= args.len() {
                return None;
            }
            let c = get(q_consts, args[sel]).and_then(Known::as_i32)?;
            let entry = targets.get(c as u32 as usize).unwrap_or(default);
            (entry.0, &entry.1)
        }
        _ => return None,
    };
    if rt == target {
        return None; // the forwarder branches to itself — leave it for prune/merge
    }
    // The forwarder defines nothing, so each resolved-edge argument is a forwarder-parameter index;
    // map it back through `args` to the value the predecessor passes for that parameter.
    let mut mapped = Vec::with_capacity(rargs.len());
    for &v in rargs {
        if (v as usize) >= args.len() {
            return None;
        }
        mapped.push(args[v as usize]);
    }
    Some((rt, mapped))
}

/// Redirect every out-edge of `term` that threads through an empty conditional forwarder
/// ([`thread_edge`]), keeping the terminator's kind; edges that do not thread are left unchanged.
fn redirect_edges(term: &Terminator, blocks: &[Block], q_consts: &[Option<Known>]) -> Terminator {
    let thread = |target: u32, args: &Vec<ValIdx>| -> (u32, Vec<ValIdx>) {
        thread_edge(target, args, blocks, q_consts).unwrap_or_else(|| (target, args.clone()))
    };
    match term {
        Terminator::Br { target, args } => {
            let (t, a) = thread(*target, args);
            Terminator::Br { target: t, args: a }
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let (tt, ta) = thread(*then_blk, then_args);
            let (et, ea) = thread(*else_blk, else_args);
            Terminator::BrIf {
                cond: *cond,
                then_blk: tt,
                then_args: ta,
                else_blk: et,
                else_args: ea,
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let targets = targets.iter().map(|(t, a)| thread(*t, a)).collect();
            let default = thread(default.0, &default.1);
            Terminator::BrTable {
                idx: *idx,
                targets,
                default,
            }
        }
        other => other.clone(),
    }
}

/// Jump threading: redirect edges that reach an empty conditional forwarder with a constant selector
/// straight to the resolved target — the classic correlated-branch simplification (`if c { … } ;
/// if c { … }`) that SCCP cannot catch, because the forwarder's selector parameter is a *different*
/// constant on each incoming edge (so its meet is not constant). Analysis reads the original blocks;
/// the surrounding fixpoint's prune/merge then cleans up any forwarder left with no predecessors, and
/// re-runs threading so multi-hop chains collapse a hop per iteration.
fn jump_thread(blocks: &[Block], fn_results: &[usize]) -> Vec<Block> {
    let consts: Vec<Vec<Option<Known>>> =
        blocks.iter().map(|b| block_consts(b, fn_results)).collect();
    blocks
        .iter()
        .enumerate()
        .map(|(q, b)| Block {
            params: b.params.clone(),
            insts: b.insts.clone(),
            term: redirect_edges(&b.term, blocks, &consts[q]),
        })
        .collect()
}

/// In a terminator, remove the edge arguments at the dropped-parameter positions of each target.
fn drop_edge_args(term: &mut Terminator, dropped: &[Vec<usize>]) {
    let trim = |args: &mut Vec<ValIdx>, target: u32| {
        for &p in dropped[target as usize].iter().rev() {
            args.remove(p);
        }
    };
    match term {
        Terminator::Br { target, args } => trim(args, *target),
        Terminator::BrIf {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            trim(then_args, *then_blk);
            trim(else_args, *else_blk);
        }
        Terminator::BrTable {
            targets, default, ..
        } => {
            for (t, args) in targets.iter_mut() {
                trim(args, *t);
            }
            trim(&mut default.1, default.0);
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => {}
    }
}
