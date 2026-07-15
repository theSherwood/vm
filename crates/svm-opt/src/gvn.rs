//! Global value numbering / CSE (see `OPT.md` Phase 2) — the first pass that moves a value **across**
//! block boundaries, and so the first consumer of the internal SSA form's cross-block-use lowering.
//!
//! The trick block-local SSA forces: two blocks never share an operand value-id (a value is used only
//! in its defining block, §3), so plain "same op + same operands" matching is inert across blocks.
//! GVN therefore computes **value-number congruence** — a block parameter is congruent to the value
//! passed to it when *every* predecessor passes a congruent value — iterated to a fixpoint. That makes
//! a recomputation in a join block congruent to the original even though its operands are fresh
//! parameters, which is exactly the redundancy `merge_blocks` + `local_cse` cannot reach (they only
//! collapse single-predecessor chains; GVN handles multi-predecessor joins / diamonds).
//!
//! A value `v` congruent to a value `L` whose definition **dominates** `v` is redundant — `L`'s value
//! reaches `v` on every path — so `v`'s uses are rewritten to `L`. Since `L` is not directly nameable
//! in `v`'s block, GVN makes it available by **threading it through block parameters** (SSA's "read a
//! variable at a block" construction), sound precisely because `L` dominates `v`. New parameters are
//! typed with the verifier's own `func_value_types`, so the threaded IR re-verifies.
//!
//! Only **pure** instructions (Phase 1a effects table) get a shared value number; a load/atomic/call
//! keeps a unique number (it may trap, read changing memory, or have effects). Runs before the
//! per-function cleanup, which DCEs the dead duplicate defs and drops any now-unused parameter.
//! Untrusted-for-escape like the rest: output re-verified.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Func, Inst, Terminator, ValType};
use svm_verify::func_value_types;

use crate::cfg::Cfg;
use crate::ssa::{from_ssa, to_ssa, Def, SsaFunc, Value};
use crate::{map_operands, map_term_operands};

/// Does block `a` dominate block `b`? Walks `b` up the immediate-dominator chain until it reaches `a`
/// (dominated) or the entry (not). `idom` is [`Cfg::dominators`] (entry / unreachable → `None`).
fn dominates(idom: &[Option<u32>], a: u32, b: u32) -> bool {
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

/// Enumerate a terminator's out-edges as `(target, args)` in canonical order.
fn edges_of(term: &Terminator) -> Vec<(u32, &Vec<Value>)> {
    match term {
        Terminator::Br { target, args } => vec![(*target, args)],
        Terminator::BrIf {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => vec![(*then_blk, then_args), (*else_blk, else_args)],
        Terminator::BrTable {
            targets, default, ..
        } => {
            let mut v: Vec<(u32, &Vec<Value>)> = targets.iter().map(|(t, a)| (*t, a)).collect();
            v.push((default.0, &default.1));
            v
        }
        _ => Vec::new(),
    }
}

/// Append `arg` to every edge of `term` that targets block `to` (a predecessor may have several: a
/// `br_if` with both arms to `to`, or a `br_table` repeating it). Keeps edge args aligned with the
/// new parameter appended at `to`.
fn append_edge_arg(term: &mut Terminator, to: u32, arg: Value) {
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
struct Threader<'a> {
    s: &'a mut SsaFunc,
    preds: &'a [Vec<u32>],
    gtype: Vec<ValType>,
    /// Memo: value `e` made available at block `b` is represented by `avail[(e, b)]`.
    avail: BTreeMap<(Value, u32), Value>,
}

impl Threader<'_> {
    /// Return a value valid **in block `at`** that equals `e` (defined in `def_b`, which dominates
    /// `at`). Adds a parameter to `at` and threads `e` in along each predecessor edge when needed.
    fn make_available(&mut self, e: Value, def_b: u32, at: u32) -> Value {
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
}

/// Run global value numbering / CSE on a function. `funcs` is the whole module's function list and
/// `has_memory` its memory presence — both only for `func_value_types` (call result types / memory
/// op validity). Semantics-preserving; meant to be followed by the ordinary cleanup, which removes
/// the dead duplicate definitions and any parameter this pass leaves unused.
pub fn gvn(f: &Func, funcs: &[Func], has_memory: bool) -> Func {
    let fn_results: Vec<usize> = funcs.iter().map(|fu| fu.results.len()).collect();
    let mut s = to_ssa(f, &fn_results);
    let nblocks = s.blocks.len();
    let nvals = s.num_values as usize;
    if nblocks == 0 || nvals == 0 {
        return from_ssa(&s);
    }

    // Per-value types, in the same block-local order as `values`, from the verifier's own typing.
    let types_local = func_value_types(f, funcs, has_memory);
    let mut gtype = vec![ValType::I32; nvals];
    for (vals, tys) in s.values.iter().zip(types_local.iter()) {
        for (&g, &ty) in vals.iter().zip(tys.iter()) {
            gtype[g as usize] = ty;
        }
    }

    let cfg = Cfg::new(&f.blocks);
    let idom = cfg.dominators();
    let rpo = cfg.rpo();

    // def_block[v] and the per-instruction result values.
    let mut def_block = vec![0u32; nvals];
    for (v, d) in s.defs.iter().enumerate() {
        def_block[v] = match d {
            Def::Param { block, .. } => *block,
            Def::Result { block, .. } => *block,
        };
    }
    let mut inst_results: Vec<Vec<Vec<Value>>> = Vec::with_capacity(nblocks);
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut per_inst = Vec::with_capacity(blk.insts.len());
        let mut slot = blk.params.len();
        for inst in &blk.insts {
            let rc = inst.result_count(&fn_results);
            per_inst.push(s.values[b][slot..slot + rc].to_vec());
            slot += rc;
        }
        inst_results.push(per_inst);
    }

    // in_args[b][j] = the values passed to block b's parameter j over *all* in-edges.
    let mut in_args: Vec<Vec<Vec<Value>>> = (0..nblocks)
        .map(|b| vec![Vec::new(); s.blocks[b].params.len()])
        .collect();
    for blk in &s.blocks {
        for (to, args) in edges_of(&blk.term) {
            for (j, &a) in args.iter().enumerate() {
                if (j) < in_args[to as usize].len() {
                    in_args[to as usize][j].push(a);
                }
            }
        }
    }

    // ---- Value-number congruence to a fixpoint. `vn[v]` is v's value number (a representative value
    // id); congruent values share it. Impure / multi-result values keep their own id (never shared).
    let mut vn: Vec<u32> = (0..nvals as u32).collect();
    // Cap iterations for termination (VN is monotone in practice; the cap is a pathology guard).
    for _ in 0..nvals + 8 {
        let mut changed = false;
        // Expression table for this sweep: canonical (operands→VN) pure instruction → its VN.
        let mut table: Vec<(Inst, u32)> = Vec::new();
        for &b in &rpo {
            let bi = b as usize;
            // Parameters: congruent to the incoming value when every predecessor agrees.
            let nparams = s.blocks[bi].params.len();
            for (&pv, incoming) in s.values[bi][..nparams].iter().zip(in_args[bi].iter()) {
                let new = if incoming.is_empty() {
                    pv // entry parameters (function inputs): unique
                } else {
                    let first = vn[incoming[0] as usize];
                    if incoming.iter().all(|&a| vn[a as usize] == first) {
                        first
                    } else {
                        pv
                    }
                };
                if new != vn[pv as usize] {
                    vn[pv as usize] = new;
                    changed = true;
                }
            }
            // Instruction results: a pure single-result op is numbered by (op, operand VNs).
            for (ii, inst) in s.blocks[bi].insts.iter().enumerate() {
                let results = &inst_results[bi][ii];
                if results.len() != 1 || !inst.effects().is_pure() {
                    continue;
                }
                let v = results[0];
                let mut canon = inst.clone();
                map_operands(&mut canon, &mut |o| vn[o as usize]);
                let new = match table.iter().find(|(c, _)| *c == canon) {
                    Some(&(_, num)) => num,
                    None => {
                        table.push((canon, v));
                        v
                    }
                };
                if new != vn[v as usize] {
                    vn[v as usize] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // ---- Redundancy: replace a value with a congruent one whose definition dominates it. Processing
    // in RPO, the first value of each VN becomes its leader; a later congruent value it dominates is
    // rewritten to it (threaded across blocks as needed). Leaders are never rewritten, so no chains.
    let mut replaced: BTreeMap<Value, Value> = BTreeMap::new();
    let mut leader: BTreeMap<u32, Value> = BTreeMap::new();
    let mut threader = Threader {
        s: &mut s,
        preds: &cfg.preds,
        gtype,
        avail: BTreeMap::new(),
    };
    // Only *instruction results* are candidates — replacing a redundant computation is the win;
    // rewriting a parameter (which merely carries a value already threaded to it) is pointless churn.
    // Parameters still received value numbers above, which is what makes the join recomputations
    // congruent in the first place.
    for &b in &rpo {
        for results in &inst_results[b as usize] {
            if results.len() != 1 {
                continue; // 0-result (no value) or multi-result (calls) — never CSE candidates
            }
            let v = results[0];
            let k = vn[v as usize];
            match leader.get(&k) {
                Some(&l)
                    if l != v && dominates(&idom, def_block[l as usize], def_block[v as usize]) =>
                {
                    let rep =
                        threader.make_available(l, def_block[l as usize], def_block[v as usize]);
                    replaced.insert(v, rep);
                }
                _ => {
                    leader.insert(k, v);
                }
            }
        }
    }

    // Rewrite every use to its representative (dead duplicate defs are cleaned by the fixpoint).
    let find = |v: Value| -> Value { replaced.get(&v).copied().unwrap_or(v) };
    for blk in threader.s.blocks.iter_mut() {
        for inst in blk.insts.iter_mut() {
            map_operands(inst, &mut |o| find(o));
        }
        map_term_operands(&mut blk.term, &mut |o| find(o));
    }

    from_ssa(&s)
}
