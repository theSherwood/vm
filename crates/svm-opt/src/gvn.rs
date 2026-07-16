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
use svm_ir::{Func, ValType};
use svm_verify::func_value_types;

use crate::cfg::Cfg;
use crate::ssa::{from_ssa, to_ssa, Def, Value};
use crate::thread::{dominates, Threader};
use crate::{map_operands, map_term_operands};

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

    // Value-number congruence (shared with LICM): congruent values share a number, which is what makes
    // a join recomputation match the original even though its operands are fresh block parameters.
    let vn = crate::vn::value_numbers(&s, &cfg, &fn_results);

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
