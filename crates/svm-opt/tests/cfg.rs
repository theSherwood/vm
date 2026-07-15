//! Unit spec for the CFG utilities (OPT.md Phase 1b): adjacency, RPO, dominators, SCCs, and
//! irreducible-aware loop headers. `Cfg` reads only terminators, so the blocks are minimal (empty
//! params/insts) and the branch selectors are placeholders.

use svm_ir::{Block, Terminator};
use svm_opt::cfg::{successors, Cfg};

fn blk(term: Terminator) -> Block {
    Block {
        params: vec![],
        insts: vec![],
        term,
    }
}
fn br(t: u32) -> Terminator {
    Terminator::Br {
        target: t,
        args: vec![],
    }
}
fn brif(a: u32, b: u32) -> Terminator {
    Terminator::BrIf {
        cond: 0,
        then_blk: a,
        then_args: vec![],
        else_blk: b,
        else_args: vec![],
    }
}
fn ret() -> Terminator {
    Terminator::Return(vec![])
}
fn cfg(terms: Vec<Terminator>) -> Cfg {
    let blocks: Vec<Block> = terms.into_iter().map(blk).collect();
    Cfg::new(&blocks)
}

#[test]
fn successors_dedup_equal_arms() {
    // A `br_if` whose arms coincide is a single successor.
    assert_eq!(successors(&brif(3, 3)), vec![3]);
    assert_eq!(successors(&brif(1, 2)), vec![1, 2]);
    assert_eq!(successors(&ret()), Vec::<u32>::new());
    let table = Terminator::BrTable {
        idx: 0,
        targets: vec![(1, vec![]), (2, vec![]), (1, vec![])],
        default: (2, vec![]),
    };
    assert_eq!(successors(&table), vec![1, 2]); // deduped, first-seen order
}

#[test]
fn straight_line_preds_succs_rpo_dom() {
    // 0 -> 1 -> 2 (return)
    let g = cfg(vec![br(1), br(2), ret()]);
    assert_eq!(g.succs, vec![vec![1], vec![2], vec![]]);
    assert_eq!(g.preds, vec![vec![], vec![0], vec![1]]);
    assert_eq!(g.rpo(), vec![0, 1, 2]);
    assert_eq!(g.postorder(), vec![2, 1, 0]);
    assert_eq!(g.dominators(), vec![None, Some(0), Some(1)]);
    assert_eq!(g.loop_headers(), vec![false, false, false]);
}

#[test]
fn diamond_dominators() {
    // 0 -> {1,2} -> 3
    let g = cfg(vec![brif(1, 2), br(3), br(3), ret()]);
    assert_eq!(g.preds[3], vec![1, 2]);
    // Every arm of the diamond is dominated directly by the entry; the join by the entry too.
    assert_eq!(g.dominators(), vec![None, Some(0), Some(0), Some(0)]);
    assert!(g.loop_headers().iter().all(|&h| !h));
}

#[test]
fn natural_loop_header() {
    // 0 -> 1 -> 2 ; 2 -> 1 (back edge) or 2 -> 3 (exit) ; 3 return
    let g = cfg(vec![br(1), br(2), brif(1, 3), ret()]);
    // SCC {1,2} is the loop; entered from 0 at block 1.
    let h = g.loop_headers();
    assert_eq!(h, vec![false, true, false, false]);
    // The loop body dominators: 1 dominates 2, entry dominates 1, 2 dominates 3.
    assert_eq!(g.dominators(), vec![None, Some(0), Some(1), Some(2)]);
    // The back edge 2->1 makes 1 a predecessor-target from inside the SCC.
    assert!(g.preds[1].contains(&2));
}

#[test]
fn irreducible_two_entry_loop_flags_both_headers() {
    // 0 -> {1,2}; 1 -> 2; 2 -> {1,3}; 3 return. The cycle {1,2} has TWO entries (0->1 and 0->2),
    // so it is irreducible — both 1 and 2 must be reported as headers.
    let g = cfg(vec![brif(1, 2), br(2), brif(1, 3), ret()]);
    // One SCC contains both 1 and 2.
    let sccs = g.sccs();
    let cyc = sccs
        .iter()
        .find(|c| c.contains(&1) && c.contains(&2))
        .expect("blocks 1 and 2 share an SCC");
    assert_eq!(cyc.len(), 2);
    assert_eq!(g.loop_headers(), vec![false, true, true, false]);
    // Dominators are still well-defined on the irreducible CFG (CHK is not reducibility-bound).
    assert_eq!(g.dominators(), vec![None, Some(0), Some(0), Some(2)]);
}

#[test]
fn self_loop_is_cyclic() {
    // 0 -> 1 ; 1 -> {1,2} ; 2 return. Block 1 self-loops.
    let g = cfg(vec![br(1), brif(1, 2), ret()]);
    assert_eq!(g.loop_headers(), vec![false, true, false]);
    let one_scc = g.sccs().into_iter().find(|c| c.contains(&1)).unwrap();
    assert_eq!(one_scc, vec![1]); // singleton, but cyclic via the self-edge
}

#[test]
fn unreachable_block_excluded_from_rpo_and_has_no_idom() {
    // 0 -> 2 (return); block 1 is unreachable (nothing branches to it).
    let g = cfg(vec![br(2), br(2), ret()]);
    assert_eq!(g.preds[1], Vec::<u32>::new());
    let rpo = g.rpo();
    assert!(!rpo.contains(&1), "unreachable block absent from RPO");
    assert_eq!(rpo, vec![0, 2]);
    assert_eq!(g.dominators()[1], None); // unreachable ⇒ no idom
    assert_eq!(g.dominators(), vec![None, None, Some(0)]);
}

#[test]
fn out_of_range_target_is_ignored_not_panicked() {
    // Fuzz safety: an unverified terminator naming a nonexistent block must not panic.
    let g = cfg(vec![br(9), ret()]);
    assert_eq!(g.succs[0], Vec::<u32>::new());
    assert_eq!(g.rpo(), vec![0]);
}
