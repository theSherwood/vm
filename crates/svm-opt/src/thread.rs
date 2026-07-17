//! Cross-block value **threading** over the internal SSA form — shared by GVN and LICM.
//!
//! When a pass wants to use a value `e` (defined in block `def_b`) inside another block `at` that
//! `def_b` dominates, block-local SSA (§3) forbids naming it directly. [`Threader::make_available`]
//! makes it nameable by adding a parameter to `at` and passing `e` along every predecessor edge,
//! recursively — SSA's "read a variable at a block" construction. It is sound precisely because
//! `def_b` dominates `at` (hence every predecessor back to `def_b`, so the value is defined on every
//! incoming edge). New parameters are typed like `e`, so the result re-verifies.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use svm_ir::{Inst, Terminator, ValType};

use crate::ssa::{SsaFunc, Value};

/// Does block `a` dominate block `b`? Walks `b` up the immediate-dominator chain until it reaches `a`
/// (dominated) or the entry (not). `idom` is [`crate::cfg::Cfg::dominators`] (entry / unreachable →
/// `None`).
pub(crate) fn dominates(idom: &[Option<u32>], a: u32, b: u32) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        match idom[x as usize] {
            Some(p) => x = p,
            None => return false,
        }
    }
}

/// Append `arg` to every edge of `term` that targets block `to` (a predecessor may have several: a
/// `br_if` with both arms to `to`, or a `br_table` repeating it). Keeps edge args aligned with the
/// new parameter appended at `to`.
pub(crate) fn append_edge_arg(term: &mut Terminator, to: u32, arg: Value) {
    match term {
        Terminator::Br { target, args } => {
            if *target == to {
                args.push(arg);
            }
        }
        Terminator::BrIf {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            if *then_blk == to {
                then_args.push(arg);
            }
            if *else_blk == to {
                else_args.push(arg);
            }
        }
        Terminator::BrTable {
            targets, default, ..
        } => {
            for (t, a) in targets.iter_mut() {
                if *t == to {
                    a.push(arg);
                }
            }
            if default.0 == to {
                default.1.push(arg);
            }
        }
        _ => {}
    }
}

/// Threads a dominating value to the blocks that need it, adding block parameters + edge args. Holds
/// the mutable SSA function plus the immutable analysis it reads (predecessors, value types).
pub(crate) struct Threader<'a> {
    pub s: &'a mut SsaFunc,
    pub preds: &'a [Vec<u32>],
    pub gtype: Vec<ValType>,
    /// Memo: value `e` made available at block `b` is represented by `avail[(e, b)]`.
    pub avail: BTreeMap<(Value, u32), Value>,
}

impl Threader<'_> {
    /// Return a value valid **in block `at`** that equals `e` (defined in `def_b`, which dominates
    /// `at`). Adds a parameter to `at` and threads `e` in along each predecessor edge when needed.
    pub fn make_available(&mut self, e: Value, def_b: u32, at: u32) -> Value {
        if at == def_b {
            return e; // e is defined right here
        }
        if let Some(&v) = self.avail.get(&(e, at)) {
            return v;
        }
        // Add a fresh parameter to `at`, typed like `e`, at the end of its parameter list.
        let ty = self.gtype[e as usize];
        let slot = self.s.blocks[at as usize].params.len();
        self.s.blocks[at as usize].params.push(ty);
        let g = self.s.num_values;
        self.s.num_values += 1;
        self.gtype.push(ty);
        self.s.values[at as usize].insert(slot, g); // new param sits right after existing params
        self.avail.insert((e, at), g); // record before recursing so cycles (loops) terminate
                                       // Feed the new parameter from every predecessor.
        for p in self.preds[at as usize].clone() {
            let arg = self.make_available(e, def_b, p);
            append_edge_arg(&mut self.s.blocks[p as usize].term, at, arg);
        }
        g
    }

    /// Append a single-result instruction to block `b` and return its fresh value, typed `ty`. Used to
    /// place a hoisted computation (or a rematerialized constant) in a preheader before threading its
    /// result back into the loop.
    pub fn emit(&mut self, b: u32, inst: Inst, ty: ValType) -> Value {
        let v = self.s.num_values;
        self.s.num_values += 1;
        self.gtype.push(ty);
        self.s.blocks[b as usize].insts.push(inst);
        self.s.values[b as usize].push(v);
        v
    }
}
