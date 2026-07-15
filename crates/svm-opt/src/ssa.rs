//! The optimizer's internal **conventional SSA** form (see `OPT.md` Phase 1c).
//!
//! The wire IR is *block-local* SSA: within a block, values are indexed `0..k` (params first, then
//! one per instruction result), and operands may name only earlier same-block indices — cross-block
//! dataflow flows exclusively through block parameters, which already act as phis (§3). That
//! discipline keeps the verifier a linear pass, but it is hostile to global passes (GVN, LICM, SCCP)
//! that must track a value across block boundaries.
//!
//! So the optimizer converts each function into this **function-global** form: the same CFG and the
//! same instructions, but every operand is a global [`Value`] id, and each id has a single recorded
//! definition site ([`Def`]). Block parameters stay as the phis they already are. Passes run here,
//! then [`from_ssa`] lowers back to block-local indices.
//!
//! **The round-trip is the identity.** [`to_ssa`] is a pure renaming (local→global) and [`from_ssa`]
//! its exact inverse (global→local), so `from_ssa(to_ssa(f)) == f` for every function — the boundary
//! is lossless before any pass touches it. That invariant is the Phase 1c deliverable, pinned by the
//! `ssa_roundtrip` unit tests and the `opt_ssa_roundtrip` fuzz target. (A pass that introduces a
//! *cross-block* use — a value referenced in a block other than its definition's — will need
//! `from_ssa` to thread it through block parameters; that lowering extension lands with the first
//! such pass in Phase 2. Today every value is used only within its defining block, so lowering is a
//! straight renaming.)

use crate::{map_operands, map_term_operands};
use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Block, Func, ValType};

/// A function-global SSA value id — the internal form's currency, versus the wire IR's block-local
/// [`svm_ir::ValIdx`].
pub type Value = u32;

/// Where a global [`Value`] is defined.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Def {
    /// A block parameter (a phi): the `param`-th parameter of `block`. Its incoming values are the
    /// matching edge args of each predecessor.
    Param { block: u32, param: u32 },
    /// The `result`-th result of the `inst`-th instruction of `block`.
    Result { block: u32, inst: u32, result: u32 },
}

/// A function in the optimizer's internal SSA form: structurally identical to the wire [`Func`],
/// **except** every operand inside `blocks` (instruction operands *and* terminator / edge args) is a
/// function-global [`Value`] id rather than a block-local [`svm_ir::ValIdx`]. Produced by
/// [`to_ssa`], lowered by [`from_ssa`].
#[derive(Clone, PartialEq, Debug)]
pub struct SsaFunc {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
    /// Blocks whose operands are **global** `Value` ids. (`Block::params` still holds the parameter
    /// *types*; a parameter's global id lives in `values[block]` and its def in `defs`.)
    pub blocks: Vec<Block>,
    /// `values[b]` = the global ids of block `b`'s local slots, in wire order (parameters first, then
    /// each instruction's results). The renaming table `from_ssa` inverts.
    pub values: Vec<Vec<Value>>,
    /// `defs[v]` = the definition site of global value `v`. `defs.len() == num_values`.
    pub defs: Vec<Def>,
    /// Total number of global values.
    pub num_values: u32,
}

/// Convert a block-local function into the internal global-SSA form. `fn_results` gives every
/// function's result arity (a `Call` appends its callee's results), exactly as [`crate::optimize_func`]
/// takes it. Pure renaming — no instruction is added, removed, or reordered.
pub fn to_ssa(f: &Func, fn_results: &[usize]) -> SsaFunc {
    // First pass: hand each local slot of each block a fresh global id, recording its def site.
    let mut values: Vec<Vec<Value>> = Vec::with_capacity(f.blocks.len());
    let mut defs: Vec<Def> = Vec::new();
    let mut next: Value = 0;
    for (b, blk) in f.blocks.iter().enumerate() {
        let mut local: Vec<Value> = Vec::with_capacity(blk.params.len() + blk.insts.len());
        for p in 0..blk.params.len() {
            local.push(next);
            defs.push(Def::Param {
                block: b as u32,
                param: p as u32,
            });
            next += 1;
        }
        for (ii, inst) in blk.insts.iter().enumerate() {
            for r in 0..inst.result_count(fn_results) {
                local.push(next);
                defs.push(Def::Result {
                    block: b as u32,
                    inst: ii as u32,
                    result: r as u32,
                });
                next += 1;
            }
        }
        values.push(local);
    }

    // Second pass: rewrite every operand from its block-local index to the block's global id.
    let mut blocks: Vec<Block> = Vec::with_capacity(f.blocks.len());
    for (b, blk) in f.blocks.iter().enumerate() {
        let g = &values[b];
        let mut nb = blk.clone();
        for inst in &mut nb.insts {
            map_operands(inst, &mut |local| g[local as usize]);
        }
        map_term_operands(&mut nb.term, &mut |local| g[local as usize]);
        blocks.push(nb);
    }

    SsaFunc {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks,
        values,
        defs,
        num_values: next,
    }
}

/// Lower the internal global-SSA form back to a block-local [`Func`]. The exact inverse of
/// [`to_ssa`] on an untransformed function, so `from_ssa(to_ssa(f)) == f`.
///
/// Every global value is defined and (today) used only within one block — the block-local invariant
/// the wire form guarantees and no current pass breaks — so inverting each block's own `values` table
/// covers every operand it references.
pub fn from_ssa(s: &SsaFunc) -> Func {
    // `inv[global] = block-local index`. A value is referenced only in its defining block, whose
    // slots we set just before remapping it, so stale entries from other blocks are never read.
    let mut inv = vec![u32::MAX; s.num_values as usize];
    let mut blocks: Vec<Block> = Vec::with_capacity(s.blocks.len());
    for (b, blk) in s.blocks.iter().enumerate() {
        for (local, &glob) in s.values[b].iter().enumerate() {
            inv[glob as usize] = local as u32;
        }
        let mut nb = blk.clone();
        for inst in &mut nb.insts {
            map_operands(inst, &mut |glob| inv[glob as usize]);
        }
        map_term_operands(&mut nb.term, &mut |glob| inv[glob as usize]);
        blocks.push(nb);
    }
    Func {
        params: s.params.clone(),
        results: s.results.clone(),
        blocks,
    }
}
