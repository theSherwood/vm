#![forbid(unsafe_code)]
//! `svm-peval` — the partial-evaluation / Futamura on-ramp (see `PEVAL.md`).
//!
//! Two layers:
//!
//! - **The generic IR→IR optimizer** ([`optimize_module`]) — Stages 0/0.x below.
//! - **The specializer** ([`specialize`]) — Stage 1, the first Futamura projection: turn an
//!   interpreter + a fixed program (in readonly "constant memory") into a compiled residual.
//!   See [`mod@specialize`].
//!
//! The optimizer is a closed-module, semantics-preserving pass that proves the
//! `rewrite → re-verify → run` loop end to end. It does four things:
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
//!
//! **Untrusted for escape (§2a / §20a posture).** Like the LLVM on-ramp, this pass is
//! *not* in the escape-TCB: its output is meant to be re-verified with
//! `svm_verify::verify_module` before it runs, so a bug here is a clean verify error, never
//! an escape. The differential harness (`tests/optimize.rs`) is the correctness spec:
//! `optimized(args) == original(args)` on the reference interpreter, for results *and*
//! traps — including a randomized fuzz over dead-heavy arithmetic DAGs that stresses the
//! renumbering/remapper.
//!
//! **Still out of scope** (later increments): float constant folding (NaN/rounding
//! fidelity); dead **block-parameter** elimination and **block merging** (collapsing a block
//! into its single unconditional predecessor) — both cross-edge transforms; and lifting an
//! interpreter's value stack out of memory into SSA so memory-backed interpreters specialize
//! too (the specializer's Stage 2).

use svm_ir::{BinOp, Block, CmpOp, ConvOp, Func, Inst, IntTy, IntUnOp, Module, Terminator, ValIdx};

mod specialize;
pub use specialize::{specialize, SpecArg, SpecError};

/// A value known to be a constant at optimization time. Tracks integers only (the only types
/// folded); floats/v128 are recorded as "unknown". Shared with the [`specialize`] engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Known {
    I32(i32),
    I64(i64),
}

impl Known {
    /// The `const` instruction that materializes this value (the in-place fold result).
    pub(crate) fn to_const_inst(self) -> Inst {
        match self {
            Known::I32(v) => Inst::ConstI32(v),
            Known::I64(v) => Inst::ConstI64(v),
        }
    }
    pub(crate) fn as_i32(self) -> Option<i32> {
        match self {
            Known::I32(v) => Some(v),
            Known::I64(_) => None,
        }
    }
    pub(crate) fn as_i64(self) -> Option<i64> {
        match self {
            Known::I64(v) => Some(v),
            Known::I32(_) => None,
        }
    }
}

/// Optimize every function in a module. Memory/data/imports are carried through unchanged;
/// `debug_info` is **dropped** because its `(func, block, inst)` positions go stale once we
/// fold instructions and drop blocks (it is strippable and untrusted for escape, §3a).
pub fn optimize_module(m: &Module) -> Module {
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    Module {
        funcs: m
            .funcs
            .iter()
            .map(|f| optimize_func(f, &fn_results))
            .collect(),
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        debug_info: None,
    }
}

/// Optimize a single function: fold + resolve branches per block, prune dead blocks, then
/// drop dead values within each surviving block. `fn_results` is the per-`FuncIdx` result
/// arity (for `Inst::result_count`).
pub fn optimize_func(f: &Func, fn_results: &[usize]) -> Func {
    let folded: Vec<Block> = f.blocks.iter().map(|b| fold_block(b, fn_results)).collect();
    let pruned = prune_unreachable(folded);
    Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks: pruned.iter().map(|b| dce_block(b, fn_results)).collect(),
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
fn const_value(inst: &Inst) -> Option<Known> {
    match *inst {
        Inst::ConstI32(v) => Some(Known::I32(v)),
        Inst::ConstI64(v) => Some(Known::I64(v)),
        _ => None,
    }
}

/// Read a block-local value as a known constant, if it is one.
fn get(known: &[Option<Known>], idx: ValIdx) -> Option<Known> {
    known.get(idx as usize).copied().flatten()
}

/// Try to fold a pure, single-result integer instruction to a constant. Returns `None` when
/// an operand is not known, the op is not foldable, or folding it would trap (div/rem by
/// zero or signed overflow) — in which case the original instruction is kept so the residual
/// traps identically to the source.
fn try_fold(inst: &Inst, known: &[Option<Known>]) -> Option<Known> {
    match *inst {
        Inst::IntBin { ty, op, a, b } => fold_int_bin(ty, op, get(known, a)?, get(known, b)?),
        Inst::IntCmp { ty, op, a, b } => fold_int_cmp(ty, op, get(known, a)?, get(known, b)?),
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
        _ => None,
    }
}

/// Constant-fold an integer binary op, mirroring the interpreter's `bin32`/`bin64` exactly
/// (wrapping arithmetic; shifts/rotates mod bitwidth). Returns `None` for the trapping cases
/// so the op is preserved and traps at runtime as the source would.
pub(crate) fn fold_int_bin(ty: IntTy, op: BinOp, a: Known, b: Known) -> Option<Known> {
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
pub(crate) fn fold_int_cmp(ty: IntTy, op: CmpOp, a: Known, b: Known) -> Option<Known> {
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
pub(crate) fn fold_int_un(ty: IntTy, op: IntUnOp, a: Known) -> Option<Known> {
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

/// Resolve a conditional terminator to an unconditional `br` when its selector is a known
/// constant, using the interpreter's exact selection rule. Non-constant selectors (and the
/// already-unconditional terminators) are returned unchanged.
fn resolve_term(t: &Terminator, known: &[Option<Known>]) -> Terminator {
    match t {
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => match get(known, *cond).and_then(Known::as_i32) {
            Some(c) if c != 0 => Terminator::Br {
                target: *then_blk,
                args: then_args.clone(),
            },
            Some(_) => Terminator::Br {
                target: *else_blk,
                args: else_args.clone(),
            },
            None => t.clone(),
        },
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => match get(known, *idx).and_then(Known::as_i32) {
            Some(c) => {
                let (target, args) = targets.get(c as u32 as usize).unwrap_or(default);
                Terminator::Br {
                    target: *target,
                    args: args.clone(),
                }
            }
            None => t.clone(),
        },
        other => other.clone(),
    }
}

/// The block successors reachable through a terminator (for the reachability walk).
fn term_successors(t: &Terminator) -> Vec<u32> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        Terminator::BrTable {
            targets, default, ..
        } => {
            let mut v: Vec<u32> = targets.iter().map(|(t, _)| *t).collect();
            v.push(default.0);
            v
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => vec![],
    }
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
/// whitelist of ops that are *pure*, *cannot trap*, and have *no side effect*, so deleting
/// one changes nothing observable. Everything else — anything that can fault (loads,
/// atomics, trapping float→int, `cap.self.get`), writes memory or state (stores, `gc.roots`),
/// transfers control / spawns / blocks (calls, `cap`/`cont`/`thread`/`memory.wait` ops,
/// fences), or is otherwise unclassified — defaults to **not** removable (kept). The
/// default direction is the safe one: a missed removal only forgoes an optimization, never
/// changes behavior.
pub fn is_removable_if_dead(inst: &Inst) -> bool {
    match inst {
        // `div`/`rem` trap on a zero (or signed-overflow) divisor; the rest of `IntBin` is pure.
        Inst::IntBin { op, .. } => !matches!(
            op,
            BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU
        ),
        Inst::ConstI32(_)
        | Inst::ConstI64(_)
        | Inst::ConstF32(_)
        | Inst::ConstF64(_)
        | Inst::ConstV128(_)
        | Inst::IntCmp { .. }
        | Inst::IntUn { .. }
        | Inst::Eqz { .. }
        | Inst::Convert { .. }
        | Inst::Select { .. }
        | Inst::FBin { .. }
        | Inst::FUn { .. }
        | Inst::FCmp { .. }
        // saturating float→int does not trap (the trapping variant, `FToITrap`, does not appear here)
        | Inst::FToISat { .. }
        | Inst::IToFConv { .. }
        | Inst::Cast { .. }
        | Inst::RefFunc { .. }
        | Inst::PtrAdd { .. }
        | Inst::PtrCast { .. }
        | Inst::SimdWidthBytes
        // all SIMD lane ops below are pure register-to-register (no memory, no trap)
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
        | Inst::VExtMul { .. }
        | Inst::VExtAddPairwise { .. }
        | Inst::VQ15MulrSat { .. }
        | Inst::VAnyTrue { .. }
        | Inst::VAllTrue { .. }
        | Inst::VBitmask { .. }
        | Inst::VFloatBin { .. }
        | Inst::VFloatUn { .. }
        | Inst::VBitBin { .. }
        | Inst::VNot { .. }
        | Inst::Bitselect { .. }
        | Inst::Shuffle { .. }
        | Inst::Swizzle { .. } => true,
        _ => false,
    }
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
        | Inst::Suspend { value: a }
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
        Inst::Bitselect { a, b, mask } => {
            *a = f(*a);
            *b = f(*b);
            *mask = f(*mask);
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
fn each_operand(inst: &Inst, mut visit: impl FnMut(ValIdx)) {
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
