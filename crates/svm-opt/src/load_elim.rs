//! Cross-block redundant-load elimination (see `OPT.md` Phase 4) — the memory analogue of GVN. A load
//! whose location was already established by a **dominating** access (an earlier load, or a matching
//! store) with **no memory write on any path between** reads the very same bytes, so it is removed and
//! its result forwarded to that earlier value — threaded across blocks exactly as GVN threads a
//! congruent dominating value.
//!
//! Block-local SSA makes "same location" subtle: two blocks never share an operand id, so the address
//! is compared by **value-number congruence** ([`crate::vn`]) — the same test GVN uses — plus the
//! access offset and op. The available value is made nameable in the load's block by
//! [`crate::thread::Threader`] (adding block parameters + edge args back to the dominating definition),
//! sound precisely because the source dominates the load.
//!
//! **Alias model (conservative across blocks).** The intra-block pass ([`crate::mem_forward`]) reasons
//! about disjoint offsets; across blocks this pass is deliberately blunt: **any** instruction that
//! writes memory or has a side effect (a store, atomic, `mem.copy`/`fill`, call — via
//! [`svm_ir::Inst::effects`]) on a between-block path *clobbers*, blocking the forward. The between
//! region is required to be **acyclic** (no loop between source and load), so the source executes once
//! before the load and the partial-block clobber checks (after the source, before the load) are valid;
//! loop-carried loads are left for a later slice.
//!
//! **Sound on value and traps.** The source dominates the load and nothing wrote memory in between, so
//! the load reads identical bytes; and it cannot fault because the dominating same-location,
//! same-width access already proved the address in-bounds (loads are confined/masked; `align` is a
//! hint). The load instruction is removed with a block-local rebuild (general DCE keeps loads as
//! possible traps, so this pass carries the safety argument). Output is re-verified.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Block, Func, Inst, ValType};
use svm_verify::func_value_types;

use crate::cfg::Cfg;
use crate::ssa::{from_ssa, to_ssa, Def, Value};
use crate::store_forward_load_op;
use crate::thread::{dominates, Threader};
use crate::{map_operands, map_term_operands};

/// A memory access: a load (a possible elimination *target*, and also a source) or a forwardable store
/// (source only). `loc` keys the location by `(address value-number, offset, load-op index)`.
struct Access {
    loc: (u32, u64, u8),
    /// The value available at the location right after this access — a load's result, or the value a
    /// store wrote.
    value: Value,
    block: u32,
    inst: u32,
    is_load: bool,
}

/// Blocks reachable from `start` over `adj` (a succ/pred adjacency), as a per-block bitmap.
fn reachable(adj: &[Vec<u32>], start: u32) -> Vec<bool> {
    let mut seen = vec![false; adj.len()];
    let mut stack = vec![start];
    seen[start as usize] = true;
    while let Some(x) = stack.pop() {
        for &y in &adj[x as usize] {
            if !seen[y as usize] {
                seen[y as usize] = true;
                stack.push(y);
            }
        }
    }
    seen
}

/// Whether the subgraph induced by the blocks in `between` (edges via `succs`) is acyclic — an
/// iterative DFS with white/gray/black colouring; a back-edge to a gray node is a cycle.
fn induced_acyclic(succs: &[Vec<u32>], between: &[bool]) -> bool {
    let n = succs.len();
    let mut color = vec![0u8; n]; // 0 = white, 1 = gray (on stack), 2 = black (done)
    for s0 in 0..n {
        if !between[s0] || color[s0] != 0 {
            continue;
        }
        let mut stack: Vec<u32> = vec![s0 as u32];
        while let Some(&node) = stack.last() {
            color[node as usize] = 1;
            let mut pushed = false;
            for &y in &succs[node as usize] {
                if !between[y as usize] {
                    continue;
                }
                match color[y as usize] {
                    1 => return false, // back-edge to a gray node → cycle
                    0 => {
                        stack.push(y);
                        pushed = true;
                        break;
                    }
                    _ => {} // black: already fully explored
                }
            }
            if !pushed {
                color[node as usize] = 2;
                stack.pop();
            }
        }
    }
    true
}

/// Whether no instruction writes memory / has a side effect on any path from the source (block `sb`,
/// after instruction `si`) to the load (block `tb`, before instruction `ti`). `between` is the acyclic
/// region between them; endpoints check only the relevant partial body, interior blocks check in full.
fn clobber_free(
    s: &crate::ssa::SsaFunc,
    between: &[bool],
    sb: u32,
    si: u32,
    tb: u32,
    ti: u32,
) -> bool {
    for (b, present) in between.iter().enumerate() {
        if !present {
            continue;
        }
        let blk = &s.blocks[b];
        let (lo, hi) = if b as u32 == sb {
            ((si + 1) as usize, blk.insts.len())
        } else if b as u32 == tb {
            (0, ti as usize)
        } else {
            (0, blk.insts.len())
        };
        for inst in &blk.insts[lo..hi] {
            let e = inst.effects();
            if e.writes_mem || e.side_effect {
                return false;
            }
        }
    }
    true
}

/// Remove the instructions at `rem` (all single-result loads whose results are already unused — their
/// uses were forwarded) from a block, renumbering its block-local values, exactly as `dce_block` packs
/// down survivors. No surviving operand references a removed load's result, so its dropped local index
/// is never looked up.
fn remove_insts(b: &Block, rem: &BTreeSet<u32>, fn_results: &[usize]) -> Block {
    let nparams = b.params.len() as u32;
    let mut result_start = Vec::with_capacity(b.insts.len());
    let mut total = nparams;
    for inst in &b.insts {
        result_start.push(total);
        total += inst.result_count(fn_results) as u32;
    }
    let mut map: Vec<Option<u32>> = vec![None; total as usize];
    for p in 0..nparams {
        map[p as usize] = Some(p);
    }
    let mut insts = Vec::with_capacity(b.insts.len());
    let mut next = nparams;
    for (i, inst) in b.insts.iter().enumerate() {
        if rem.contains(&(i as u32)) {
            continue; // dropped load: results left unmapped (provably unused)
        }
        let mut ni = inst.clone();
        map_operands(&mut ni, &mut |o| {
            map[o as usize].expect("operand defined before use")
        });
        let rc = inst.result_count(fn_results) as u32;
        for r in 0..rc {
            map[(result_start[i] + r) as usize] = Some(next);
            next += 1;
        }
        insts.push(ni);
    }
    let mut term = b.term.clone();
    map_term_operands(&mut term, &mut |o| {
        map[o as usize].expect("terminator operand defined")
    });
    Block {
        params: b.params.clone(),
        insts,
        term,
    }
}

/// Run cross-block redundant-load elimination. `funcs` / `has_memory` are only for `func_value_types`
/// (typing the threaded parameters). Semantics-preserving; the surrounding cleanup drops any parameter
/// this pass threads through but leaves unused.
pub fn load_elim(f: &Func, funcs: &[Func], has_memory: bool) -> Func {
    let fn_results: Vec<usize> = funcs.iter().map(|fu| fu.results.len()).collect();
    let s = to_ssa(f, &fn_results);
    let nblocks = s.blocks.len();
    if nblocks == 0 {
        return from_ssa(&s);
    }
    let cfg = Cfg::new(&f.blocks);
    let idom = cfg.dominators();
    let rpo = cfg.rpo();
    let vn = crate::vn::value_numbers(&s, &cfg, &fn_results);
    let nvals = s.num_values as usize;

    let mut def_block = vec![0u32; nvals];
    for (v, d) in s.defs.iter().enumerate() {
        def_block[v] = match d {
            Def::Param { block, .. } | Def::Result { block, .. } => *block,
        };
    }

    // Collect every memory access (scalar load / forwardable store) with its location key + value.
    let mut accesses: Vec<Access> = Vec::new();
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut slot = blk.params.len();
        for (ii, inst) in blk.insts.iter().enumerate() {
            match inst {
                Inst::Load {
                    op, addr, offset, ..
                } => accesses.push(Access {
                    loc: (vn[*addr as usize], *offset, op.index()),
                    value: s.values[b][slot], // the load's single result
                    block: b as u32,
                    inst: ii as u32,
                    is_load: true,
                }),
                Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                    ..
                } => {
                    if let Some(lop) = store_forward_load_op(*op) {
                        accesses.push(Access {
                            loc: (vn[*addr as usize], *offset, lop.index()),
                            value: *value, // the stored value (global id)
                            block: b as u32,
                            inst: ii as u32,
                            is_load: false,
                        });
                    }
                }
                _ => {}
            }
            slot += inst.result_count(&fn_results);
        }
    }

    // Targets = loads, visited in RPO (a dominating source is decided before the loads it dominates;
    // a load already chosen as a target is never reused as a source, so no forwarding chains form).
    let mut rpo_pos = vec![0u32; nblocks];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_pos[b as usize] = i as u32;
    }
    let mut targets: Vec<usize> = (0..accesses.len())
        .filter(|&i| accesses[i].is_load)
        .collect();
    targets.sort_by_key(|&i| (rpo_pos[accesses[i].block as usize], accesses[i].inst));

    // (target value, target block, target inst, source value) for each eliminable load.
    let mut redundant: Vec<(Value, u32, u32, Value)> = Vec::new();
    let mut eliminated: BTreeSet<Value> = BTreeSet::new();
    for &ti in &targets {
        let (t_loc, t_value, t_block, t_inst) = {
            let t = &accesses[ti];
            (t.loc, t.value, t.block, t.inst)
        };
        for src in &accesses {
            if src.loc != t_loc || src.block == t_block || src.value == t_value {
                continue; // must be a different cross-block access to the same location
            }
            if src.is_load && eliminated.contains(&src.value) {
                continue; // don't chain off a load we are already removing
            }
            if !dominates(&idom, src.block, t_block) {
                continue;
            }
            let between: Vec<bool> = {
                let fwd = reachable(&cfg.succs, src.block);
                let bwd = reachable(&cfg.preds, t_block);
                fwd.iter().zip(&bwd).map(|(&a, &b)| a && b).collect()
            };
            if !induced_acyclic(&cfg.succs, &between) {
                continue; // a loop between them — partial-block reasoning would be unsound
            }
            if !clobber_free(&s, &between, src.block, src.inst, t_block, t_inst) {
                continue;
            }
            eliminated.insert(t_value);
            redundant.push((t_value, t_block, t_inst, src.value));
            break;
        }
    }

    if redundant.is_empty() {
        return from_ssa(&s);
    }

    // Thread each source value into the load's block and forward the load's uses to it.
    let types_local = func_value_types(f, funcs, has_memory);
    let mut gtype = vec![ValType::I32; nvals];
    for (vals, tys) in s.values.iter().zip(types_local.iter()) {
        for (&g, &ty) in vals.iter().zip(tys.iter()) {
            gtype[g as usize] = ty;
        }
    }
    let mut s = s;
    let mut replaced: BTreeMap<Value, Value> = BTreeMap::new();
    {
        let mut threader = Threader {
            s: &mut s,
            preds: &cfg.preds,
            gtype,
            avail: BTreeMap::new(),
        };
        for &(tv, tb, _ti, sv) in &redundant {
            let rep = threader.make_available(sv, def_block[sv as usize], tb);
            replaced.insert(tv, rep);
        }
    }
    let find = |v: Value| -> Value { replaced.get(&v).copied().unwrap_or(v) };
    for blk in s.blocks.iter_mut() {
        for inst in blk.insts.iter_mut() {
            map_operands(inst, &mut |o| find(o));
        }
        map_term_operands(&mut blk.term, &mut |o| find(o));
    }

    // Lower back, then remove the now-unused redundant loads (block+inst indices are stable across the
    // SSA round-trip and threading, which only add parameters).
    let f2 = from_ssa(&s);
    let mut by_block: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &(_tv, tb, ti, _sv) in &redundant {
        by_block.entry(tb).or_default().insert(ti);
    }
    let mut blocks = f2.blocks;
    for (b, rem) in by_block {
        blocks[b as usize] = remove_insts(&blocks[b as usize], &rem, &fn_results);
    }
    Func {
        params: f2.params,
        results: f2.results,
        blocks,
    }
}
