//! Value-number **congruence** over the internal SSA form — the shared analysis behind GVN and LICM.
//!
//! Two values get the same number iff they provably compute the same thing: a pure single-result op
//! is numbered by `(op, operand numbers)`; a block parameter is congruent to the value passed to it
//! when *every* predecessor passes a congruent value (iterated to a fixpoint). Impure / multi-result
//! values keep their own unique number (never shared). The number assigned to a congruence class is
//! the id of its first member in RPO — a stable representative.

use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Inst, Terminator};

use crate::cfg::Cfg;
use crate::map_operands;
use crate::ssa::{SsaFunc, Value};

/// Enumerate a terminator's out-edges as `(target, args)`.
fn edges(term: &Terminator) -> Vec<(u32, &Vec<Value>)> {
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

/// Compute a value number for every value of `s` (indexed by [`Value`]). Congruent values share a
/// number; see the module docs.
pub fn value_numbers(s: &SsaFunc, cfg: &Cfg, fn_results: &[usize]) -> Vec<u32> {
    let nblocks = s.blocks.len();
    let nvals = s.num_values as usize;
    let rpo = cfg.rpo();

    // inst_results[b][ii] = the value ids the ii-th instruction of block b defines.
    let mut inst_results: Vec<Vec<Vec<Value>>> = Vec::with_capacity(nblocks);
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut per = Vec::with_capacity(blk.insts.len());
        let mut slot = blk.params.len();
        for inst in &blk.insts {
            let rc = inst.result_count(fn_results);
            per.push(s.values[b][slot..slot + rc].to_vec());
            slot += rc;
        }
        inst_results.push(per);
    }

    // in_args[b][j] = the values passed to block b's parameter j over all in-edges.
    let mut in_args: Vec<Vec<Vec<Value>>> = (0..nblocks)
        .map(|b| vec![Vec::new(); s.blocks[b].params.len()])
        .collect();
    for blk in &s.blocks {
        for (to, args) in edges(&blk.term) {
            for (j, &a) in args.iter().enumerate() {
                if j < in_args[to as usize].len() {
                    in_args[to as usize][j].push(a);
                }
            }
        }
    }

    let mut vn: Vec<u32> = (0..nvals as u32).collect();
    // VN is monotone in practice; the cap is a pathology guard for termination.
    for _ in 0..nvals + 8 {
        let mut changed = false;
        let mut table: Vec<(Inst, u32)> = Vec::new(); // canonical (operands→VN) pure inst → its VN
        for &b in &rpo {
            let bi = b as usize;
            let nparams = s.blocks[bi].params.len();
            // Parameters: congruent to the incoming value when every predecessor agrees.
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
    vn
}
