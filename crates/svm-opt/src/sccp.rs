//! Sparse Conditional Constant Propagation (see `OPT.md` Phase 2).
//!
//! The existing cleanup ([`crate::optimize_func`]) folds constants *within* a block and resolves a
//! branch once its selector is a same-block constant. SCCP does it **globally**: it propagates a
//! constant lattice across the whole CFG — through block parameters (the phis) and around loops —
//! while simultaneously discovering which edges are executable, so a value is only marked *varying*
//! on account of edges that can actually be taken. That two-in-one (constants + reachability) is what
//! lets it fold things the per-block pass cannot: a join value that is the same constant on every
//! reachable predecessor, a loop-invariant constant, a branch whose selector only becomes constant
//! after cross-block propagation.
//!
//! It runs on the internal SSA form ([`crate::ssa`]) so values have global names and uses are easy to
//! chase. The transfer function **reuses [`crate::try_fold`]** — the exact, interpreter-matching fold
//! — so a value is `Const` only when the reference interpreter would compute that same constant, and
//! a trapping/effectful op is never marked constant. The rewrite is deliberately minimal: replace
//! each proven-constant instruction result with its `const`, and resolve each constant-selector
//! branch — then hand off to the existing fixpoint, which prunes the now-unreachable blocks, DCEs the
//! dead selector computations, merges, and re-folds. Because SCCP only ever *materializes constants*
//! (never moves a value between blocks), lowering back needs no block-parameter threading.
//!
//! Untrusted-for-escape like the rest of the optimizer: the output is re-verified before it runs, and
//! the differential harness (`tests/sccp.rs`) is the correctness spec — `sccp`'d module ≡ original on
//! the reference interpreter, for results *and* traps *and* final memory.

use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Func, Inst, Terminator};

use crate::ssa::{from_ssa, to_ssa, Value};
use crate::{const_value, each_operand, resolve_term, try_fold, Known};

/// The constant lattice: `Top` (not yet known — may still become constant) ⊒ `Const(k)` ⊒ `Bottom`
/// (overdefined / varies at runtime). Monotone: a value only ever moves *down*.
#[derive(Clone, Copy, PartialEq)]
enum Lat {
    Top,
    Const(Known),
    Bottom,
}

impl Lat {
    /// Greatest lower bound. `Top` is the identity; `Bottom` is absorbing; two constants agree only
    /// if equal (else the meet varies).
    fn meet(self, other: Lat) -> Lat {
        match (self, other) {
            (Lat::Top, x) | (x, Lat::Top) => x,
            (Lat::Bottom, _) | (_, Lat::Bottom) => Lat::Bottom,
            (Lat::Const(a), Lat::Const(b)) => {
                if a == b {
                    Lat::Const(a)
                } else {
                    Lat::Bottom
                }
            }
        }
    }
}

/// One CFG edge, carrying the block-argument values it passes to its target's parameters.
struct Edge {
    to: u32,
    args: Vec<Value>,
}

/// Enumerate a terminator's out-edges in a canonical order — `br`: [target]; `br_if`: [then, else];
/// `br_table`: [targets…, default]; returns/tail-calls/`unreachable`: none. The order matches
/// [`selector_edges`] so an index means the same edge in both.
fn term_edges(t: &Terminator) -> Vec<Edge> {
    match t {
        Terminator::Br { target, args } => vec![Edge {
            to: *target,
            args: args.clone(),
        }],
        Terminator::BrIf {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => vec![
            Edge {
                to: *then_blk,
                args: then_args.clone(),
            },
            Edge {
                to: *else_blk,
                args: else_args.clone(),
            },
        ],
        Terminator::BrTable {
            targets, default, ..
        } => {
            let mut v: Vec<Edge> = targets
                .iter()
                .map(|(to, args)| Edge {
                    to: *to,
                    args: args.clone(),
                })
                .collect();
            v.push(Edge {
                to: default.0,
                args: default.1.clone(),
            });
            v
        }
        _ => Vec::new(),
    }
}

/// Which out-edge indices (into [`term_edges`] order) are executable given the terminator's selector
/// lattice. A `Const` selector picks exactly one; a `Bottom` selector enables all; a `Top` selector
/// enables none yet (its edges wait until the selector resolves).
fn selector_edges(t: &Terminator, lat: &[Lat], n_edges: usize) -> Vec<usize> {
    let all = || (0..n_edges).collect::<Vec<_>>();
    match t {
        Terminator::Br { .. } => vec![0],
        Terminator::BrIf { cond, .. } => match lat[*cond as usize] {
            Lat::Const(k) => match k.as_i32() {
                Some(c) if c != 0 => vec![0],
                Some(_) => vec![1],
                None => all(), // wrong-typed constant (never on verified IR): stay conservative
            },
            Lat::Bottom => all(),
            Lat::Top => Vec::new(),
        },
        Terminator::BrTable { idx, targets, .. } => match lat[*idx as usize] {
            Lat::Const(k) => match k.as_i32() {
                Some(c) => {
                    let sel = (c as u32 as usize).min(targets.len()); // out of range ⇒ default (index = targets.len())
                    vec![sel]
                }
                None => all(),
            },
            Lat::Bottom => all(),
            Lat::Top => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Run SCCP on a function and return the rewritten function (constants materialized, constant
/// branches resolved). Semantics-preserving; meant to be followed by the ordinary cleanup fixpoint.
pub fn sccp(f: &Func, fn_results: &[usize]) -> Func {
    let mut s = to_ssa(f, fn_results);
    let nblocks = s.blocks.len();
    if nblocks == 0 {
        return from_ssa(&s);
    }
    let nvals = s.num_values as usize;

    // ---- Precompute the edge graph and the per-instruction result values. ----
    let mut edges: Vec<Edge> = Vec::new();
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); nblocks]; // out_edges[b][k] = eid of b's k-th edge
    for (b, blk) in s.blocks.iter().enumerate() {
        for e in term_edges(&blk.term) {
            let eid = edges.len();
            out_edges[b].push(eid);
            edges.push(e);
        }
    }
    // inst_results[b][ii] = the global value ids the ii-th instruction of block b defines.
    let mut inst_results: Vec<Vec<Vec<Value>>> = Vec::with_capacity(nblocks);
    for (b, blk) in s.blocks.iter().enumerate() {
        let mut per_inst = Vec::with_capacity(blk.insts.len());
        let mut slot = blk.params.len();
        for inst in &blk.insts {
            let rc = inst.result_count(fn_results);
            per_inst.push(s.values[b][slot..slot + rc].to_vec());
            slot += rc;
        }
        inst_results.push(per_inst);
    }

    // ---- Use lists (who must be re-evaluated when a value's lattice drops). ----
    let mut inst_uses: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nvals]; // value -> (block, inst_idx)
    let mut sel_uses: Vec<Vec<u32>> = vec![Vec::new(); nvals]; // value -> blocks using it as a selector
    let mut arg_targets: Vec<Vec<u32>> = vec![Vec::new(); nvals]; // value -> target blocks it is an edge-arg for
    for (b, blk) in s.blocks.iter().enumerate() {
        for (ii, inst) in blk.insts.iter().enumerate() {
            each_operand(inst, |v| inst_uses[v as usize].push((b as u32, ii as u32)));
        }
        match &blk.term {
            Terminator::BrIf { cond, .. } => sel_uses[*cond as usize].push(b as u32),
            Terminator::BrTable { idx, .. } => sel_uses[*idx as usize].push(b as u32),
            _ => {}
        }
        for e in &out_edges[b] {
            let edge = &edges[*e];
            for &v in &edge.args {
                arg_targets[v as usize].push(edge.to);
            }
        }
    }

    // ---- SCCP state. ----
    let mut lat = vec![Lat::Top; nvals];
    let mut known: Vec<Option<Known>> = vec![None; nvals]; // the `Const` view, for try_fold / resolve_term
    let mut reachable = vec![false; nblocks];
    let mut edge_exec = vec![false; edges.len()];
    let mut flow_wl: Vec<usize> = Vec::new(); // edge ids
    let mut ssa_wl: Vec<Value> = Vec::new();

    // Lower a value's lattice (monotone) and, on a change, remember the new `Const` view and re-queue.
    macro_rules! set_val {
        ($v:expr, $new:expr) => {{
            let v = $v as usize;
            let merged = lat[v].meet($new);
            if merged != lat[v] {
                lat[v] = merged;
                known[v] = if let Lat::Const(k) = merged {
                    Some(k)
                } else {
                    None
                };
                ssa_wl.push(v as Value);
            }
        }};
    }

    // Transfer function for a single-result instruction (its result value's new lattice).
    let eval_inst = |inst: &Inst, lat: &[Lat], known: &[Option<Known>]| -> Lat {
        if let Some(k) = const_value(inst) {
            return Lat::Const(k); // a literal `const`
        }
        let mut any_top = false;
        let mut any_bottom = false;
        each_operand(inst, |v| match lat[v as usize] {
            Lat::Top => any_top = true,
            Lat::Bottom => any_bottom = true,
            Lat::Const(_) => {}
        });
        if any_bottom {
            return Lat::Bottom; // a varying operand ⇒ result varies
        }
        if any_top {
            return Lat::Top; // wait for operands to resolve
        }
        // All operands constant: fold exactly like the interpreter, or mark varying if it can't fold
        // (a trapping op, a load, a call — anything `try_fold` declines).
        match try_fold(inst, known) {
            Some(k) => Lat::Const(k),
            None => Lat::Bottom,
        }
    };

    // Evaluate block b's parameters (phis): each is the meet of the matching arg over executable
    // in-edges. The entry block's parameters are the function's inputs → `Bottom`.
    macro_rules! eval_params {
        ($b:expr) => {{
            let b = $b as usize;
            let nparams = s.blocks[b].params.len();
            for j in 0..nparams {
                let pv = s.values[b][j];
                if b == 0 {
                    set_val!(pv, Lat::Bottom);
                    continue;
                }
                let mut acc = Lat::Top;
                for eid in 0..edges.len() {
                    if edge_exec[eid] && edges[eid].to as usize == b {
                        let a = edges[eid].args.get(j).copied();
                        let l = a.map(|v| lat[v as usize]).unwrap_or(Lat::Bottom);
                        acc = acc.meet(l);
                    }
                }
                set_val!(pv, acc);
            }
        }};
    }

    // Evaluate a terminator: queue every out-edge its selector currently makes executable.
    macro_rules! eval_term {
        ($b:expr) => {{
            let b = $b as usize;
            let t = &s.blocks[b].term;
            for k in selector_edges(t, &lat, out_edges[b].len()) {
                let eid = out_edges[b][k];
                if !edge_exec[eid] {
                    flow_wl.push(eid);
                }
            }
        }};
    }

    // Evaluate every instruction of a block (its results), then its terminator.
    macro_rules! eval_body {
        ($b:expr) => {{
            let b = $b as usize;
            for ii in 0..s.blocks[b].insts.len() {
                let results = &inst_results[b][ii];
                if results.len() == 1 {
                    let nl = eval_inst(&s.blocks[b].insts[ii], &lat, &known);
                    set_val!(results[0], nl);
                } else {
                    for &r in results {
                        set_val!(r, Lat::Bottom); // 0-result (no value) skipped; multi-result ⇒ varies
                    }
                }
            }
            eval_term!(b);
        }};
    }

    // ---- Seed: the entry block is reachable. ----
    reachable[0] = true;
    eval_params!(0u32);
    eval_body!(0u32);

    // ---- Fixed point over the two worklists. ----
    loop {
        if let Some(eid) = flow_wl.pop() {
            if edge_exec[eid] {
                continue; // already processed; arg-driven param updates ride the SSA worklist
            }
            edge_exec[eid] = true;
            let to = edges[eid].to;
            eval_params!(to);
            if !reachable[to as usize] {
                reachable[to as usize] = true;
                eval_body!(to);
            }
            continue;
        }
        if let Some(v) = ssa_wl.pop() {
            for &(b, ii) in &inst_uses[v as usize].clone() {
                let results = &inst_results[b as usize][ii as usize];
                if results.len() == 1 {
                    let nl = eval_inst(&s.blocks[b as usize].insts[ii as usize], &lat, &known);
                    set_val!(results[0], nl);
                }
            }
            for &b in &sel_uses[v as usize].clone() {
                eval_term!(b);
            }
            for &b in &arg_targets[v as usize].clone() {
                if reachable[b as usize] {
                    eval_params!(b);
                }
            }
            continue;
        }
        break;
    }

    // ---- Rewrite: materialize proven constants, resolve constant-selector branches. ----
    for (b, blk) in s.blocks.iter_mut().enumerate() {
        for (ii, inst) in blk.insts.iter_mut().enumerate() {
            let results = &inst_results[b][ii];
            if results.len() == 1 {
                if let Some(k) = known[results[0] as usize] {
                    *inst = k.to_const_inst();
                }
            }
        }
        blk.term = resolve_term(&blk.term, &known);
    }

    from_ssa(&s)
}
