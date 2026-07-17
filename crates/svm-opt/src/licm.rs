//! Loop-invariant code motion (see `OPT.md` Phase 2). Hoists a pure, non-trapping computation out of
//! a loop when its value does not change across iterations, so it runs once instead of every time
//! around — the biggest interp-path win for loop-heavy code (the interpreter re-executes the body
//! every iteration).
//!
//! Block-local SSA makes invariance subtle: an instruction's operands are always same-block, so a
//! value is loop-invariant only *through* block-parameter phis. We compute it the classic way,
//! iteratively: a value defined outside the loop is invariant; a loop block **parameter** is invariant
//! when every incoming argument is either already invariant or is the parameter itself (the
//! `x` passes `x` unchanged around the back edge — the archetypal invariant); a pure op is invariant
//! when all its operands are.
//!
//! An invariant instruction in the loop is hoisted by **cloning** it into the loop's preheader (the
//! unique out-of-loop predecessor of the header) and **threading** the clone's result back into the
//! loop ([`crate::thread`]); the original then has no uses and the cleanup DCEs it. Operands are
//! rewritten to values valid in the preheader — a value already dominating the preheader is used as
//! is; an invariant header parameter becomes the value the preheader passes it (its entry argument).
//!
//! Sound by construction: hoisting a **pure, non-trapping** op to the preheader (which dominates the
//! whole loop) is observationally identical even though it now runs unconditionally — it cannot trap
//! or have effects, and its result is used only where it was before. Only reducible loops (a single
//! SCC header) with a real, unique preheader are touched; anything else, or any operand that cannot be
//! named in the preheader, is left alone. Single sweep; the surrounding pipeline re-runs it. Output is
//! re-verified.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Func, Inst, Terminator, ValType};
use svm_verify::func_value_types;

use crate::cfg::Cfg;
use crate::ssa::{from_ssa, to_ssa, Def, Value};
use crate::thread::{dominates, Threader};
use crate::{each_operand, map_operands, map_term_operands};

/// How an invariant operand is named in the preheader: either an existing value that already reaches
/// the preheader, or a constant to **rematerialize** there (a fresh copy) rather than thread in.
enum Rep {
    Value(Value),
    Const(Inst),
}

/// One hoist to apply: the loop instruction's result `rv` (in `block`, its original form `inst`) moves
/// to preheader `ph`, typed `ty`. `reps` names each operand in the preheader; operands are rewritten
/// when the hoist is applied (after any constants are materialized there).
struct Hoist {
    rv: Value,
    block: u32,
    ph: u32,
    inst: Inst,
    reps: BTreeMap<Value, Rep>,
    ty: ValType,
}

/// A rematerializable constant: recomputing it anywhere is free, so hoisting one out of a loop only
/// adds a threaded block parameter (pure overhead). Such ops are never hoisted on their own; when a
/// worthwhile hoist *uses* one, it is re-emitted in the preheader instead of threaded.
fn is_const(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::ConstI32(_)
            | Inst::ConstI64(_)
            | Inst::ConstF32(_)
            | Inst::ConstF64(_)
            | Inst::ConstV128(_)
    )
}

/// The result type of a constant instruction (used to type a rematerialized copy). Only called on
/// values for which [`is_const`] holds.
fn const_type(inst: &Inst) -> ValType {
    match inst {
        Inst::ConstI32(_) => ValType::I32,
        Inst::ConstI64(_) => ValType::I64,
        Inst::ConstF32(_) => ValType::F32,
        Inst::ConstF64(_) => ValType::F64,
        Inst::ConstV128(_) => ValType::V128,
        _ => unreachable!("const_type on a non-constant instruction"),
    }
}

/// Enumerate a terminator's out-edges as `(target, &args)`.
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

/// Run loop-invariant code motion. `funcs` / `has_memory` are only for `func_value_types` (to type
/// the threaded parameters). Semantics-preserving; the cleanup fixpoint DCEs the emptied-out
/// originals.
pub fn licm(f: &Func, funcs: &[Func], has_memory: bool) -> Func {
    let fn_results: Vec<usize> = funcs.iter().map(|fu| fu.results.len()).collect();
    let mut s = to_ssa(f, &fn_results);
    let nblocks = s.blocks.len();
    if nblocks == 0 {
        return from_ssa(&s);
    }

    let cfg = Cfg::new(&f.blocks);
    let idom = cfg.dominators();
    let nvals = s.num_values as usize;

    // Per-value types (for the threaded parameters) and definition blocks.
    let types_local = func_value_types(f, funcs, has_memory);
    let mut gtype = vec![ValType::I32; nvals];
    for (vals, tys) in s.values.iter().zip(types_local.iter()) {
        for (&g, &ty) in vals.iter().zip(tys.iter()) {
            gtype[g as usize] = ty;
        }
    }
    let mut def_block = vec![0u32; nvals];
    for (v, d) in s.defs.iter().enumerate() {
        def_block[v] = match d {
            Def::Param { block, .. } | Def::Result { block, .. } => *block,
        };
    }
    // Values defined by a constant instruction — rematerializable in the preheader instead of threaded.
    let mut const_def: BTreeMap<Value, Inst> = BTreeMap::new();
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut slot = blk.params.len();
        for inst in &blk.insts {
            if is_const(inst) {
                const_def.insert(s.values[b][slot], inst.clone());
            }
            slot += inst.result_count(&fn_results);
        }
    }
    // inst_results[b][ii] and in_args[b][j] (all incoming args for parameter j of block b).
    let mut inst_results: Vec<Vec<Vec<Value>>> = Vec::with_capacity(nblocks);
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut per = Vec::with_capacity(blk.insts.len());
        let mut slot = blk.params.len();
        for inst in &blk.insts {
            let rc = inst.result_count(&fn_results);
            per.push(s.values[b][slot..slot + rc].to_vec());
            slot += rc;
        }
        inst_results.push(per);
    }
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

    // ---- Phase A: find hoists (immutable analysis). ----
    let sccs = cfg.sccs();
    let headers = cfg.loop_headers();
    let mut hoists: Vec<Hoist> = Vec::new();
    for comp in &sccs {
        let cyclic =
            comp.len() > 1 || (comp.len() == 1 && cfg.succs[comp[0] as usize].contains(&comp[0]));
        if !cyclic {
            continue;
        }
        // Reducible only: exactly one SCC block is entered from outside.
        let hdrs: Vec<u32> = comp
            .iter()
            .copied()
            .filter(|&b| headers[b as usize])
            .collect();
        if hdrs.len() != 1 {
            continue;
        }
        let h = hdrs[0];
        let loop_set: BTreeSet<u32> = comp.iter().copied().collect();

        // Unique out-of-loop predecessor = the preheader.
        let ext_preds: Vec<u32> = cfg.preds[h as usize]
            .iter()
            .copied()
            .filter(|p| !loop_set.contains(p))
            .collect();
        if ext_preds.len() != 1 {
            continue; // no clean preheader
        }
        let ph = ext_preds[0];

        // Header parameter → the value the preheader passes it (its entry argument).
        let nhp = s.blocks[h as usize].params.len();
        let mut entry_arg: Vec<Option<Value>> = vec![None; nhp];
        for (to, args) in edges(&s.blocks[ph as usize].term) {
            if to == h {
                for (j, &a) in args.iter().enumerate() {
                    if j < nhp {
                        entry_arg[j] = Some(a);
                    }
                }
            }
        }
        // header_slot[v] = Some(j) iff v is header parameter j.
        let mut header_slot: BTreeMap<Value, usize> = BTreeMap::new();
        for (j, &pv) in s.values[h as usize][..nhp].iter().enumerate() {
            header_slot.insert(pv, j);
        }

        // Iterative loop-invariance over the loop's values.
        let mut invariant = vec![false; nvals];
        for v in 0..nvals {
            if !loop_set.contains(&def_block[v]) {
                invariant[v] = true; // defined outside the loop
            }
        }
        loop {
            let mut changed = false;
            for &b in comp {
                let bi = b as usize;
                let np = s.blocks[bi].params.len();
                // Parameters: invariant if every incoming arg is invariant or the parameter itself.
                for (j, &pv) in s.values[bi][..np].iter().enumerate() {
                    if invariant[pv as usize] {
                        continue;
                    }
                    if in_args[bi][j]
                        .iter()
                        .all(|&a| a == pv || invariant[a as usize])
                    {
                        invariant[pv as usize] = true;
                        changed = true;
                    }
                }
                // Pure single-result instructions: invariant if all operands are.
                for (ii, results) in inst_results[bi].iter().enumerate() {
                    if results.len() != 1 || invariant[results[0] as usize] {
                        continue;
                    }
                    let inst = &s.blocks[bi].insts[ii];
                    if !inst.effects().is_pure() {
                        continue;
                    }
                    let mut all_inv = true;
                    each_operand(inst, |o| {
                        if !invariant[o as usize] {
                            all_inv = false;
                        }
                    });
                    if all_inv {
                        invariant[results[0] as usize] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Name an invariant operand in the preheader: a value already dominating the preheader is used
        // directly; a loop-body constant is rematerialized there; an invariant header parameter becomes
        // its entry argument. Otherwise not hoistable.
        let avail_at_ph = |o: Value| -> Option<Rep> {
            if dominates(&idom, def_block[o as usize], ph) {
                return Some(Rep::Value(o));
            }
            if let Some(c) = const_def.get(&o) {
                return Some(Rep::Const(c.clone()));
            }
            if let Some(&j) = header_slot.get(&o) {
                if invariant[o as usize] {
                    return entry_arg[j].map(Rep::Value);
                }
            }
            None
        };

        for &b in comp {
            let bi = b as usize;
            for (ii, results) in inst_results[bi].iter().enumerate() {
                if results.len() != 1 {
                    continue;
                }
                let inst = &s.blocks[bi].insts[ii];
                if !inst.effects().is_pure() {
                    continue; // must be pure + non-trapping to speculate above the loop
                }
                if is_const(inst) {
                    continue; // free to recompute — threading it out would be pure overhead
                }
                // Every operand must be nameable in the preheader.
                let mut reps: BTreeMap<Value, Rep> = BTreeMap::new();
                let mut ok = true;
                each_operand(inst, |o| match avail_at_ph(o) {
                    Some(r) => {
                        reps.insert(o, r);
                    }
                    None => ok = false,
                });
                if !ok {
                    continue;
                }
                let rv = results[0];
                hoists.push(Hoist {
                    rv,
                    block: b,
                    ph,
                    inst: inst.clone(),
                    reps,
                    ty: gtype[rv as usize],
                });
            }
        }
    }

    if hoists.is_empty() {
        return from_ssa(&s);
    }

    // ---- Phase B: apply hoists (clone into the preheader, thread the result back into the loop). ----
    let mut threader = Threader {
        s: &mut s,
        preds: &cfg.preds,
        gtype,
        avail: BTreeMap::new(),
    };
    // Constants rematerialized in a preheader, keyed per preheader, so each distinct constant is
    // emitted at most once there.
    let mut const_cache: BTreeMap<u32, Vec<(Inst, Value)>> = BTreeMap::new();
    for h in hoists {
        // Name every operand in the preheader, materializing constants (deduped per preheader).
        let mut operand_val: BTreeMap<Value, Value> = BTreeMap::new();
        for (&o, rep) in &h.reps {
            let v = match rep {
                Rep::Value(v) => *v,
                Rep::Const(c) => {
                    let cache = const_cache.entry(h.ph).or_default();
                    if let Some((_, v)) = cache.iter().find(|(i, _)| i == c) {
                        *v
                    } else {
                        let v = threader.emit(h.ph, c.clone(), const_type(c));
                        const_cache.entry(h.ph).or_default().push((c.clone(), v));
                        v
                    }
                }
            };
            operand_val.insert(o, v);
        }
        let mut clone = h.inst.clone();
        map_operands(&mut clone, &mut |o| operand_val[&o]);
        let new_rv = threader.emit(h.ph, clone, h.ty);
        let threaded = threader.make_available(new_rv, h.ph, h.block);
        let bi = h.block as usize;
        for inst in &mut threader.s.blocks[bi].insts {
            map_operands(inst, &mut |o| if o == h.rv { threaded } else { o });
        }
        map_term_operands(&mut threader.s.blocks[bi].term, &mut |o| {
            if o == h.rv {
                threaded
            } else {
                o
            }
        });
        // The original instruction now has no uses; the cleanup fixpoint DCEs it (pure + non-trapping).
    }

    from_ssa(&s)
}
