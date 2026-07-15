//! CFG analysis utilities over a function's block-local IR (see `OPT.md` Phase 1b).
//!
//! The IR supports **irreducible** control flow natively (no relooper, §3), so nothing here may
//! assume reducibility: [`Cfg::dominators`] is the iterative Cooper–Harvey–Kennedy solver (correct on
//! any CFG), and loops are found via [`Cfg::sccs`] (Tarjan) so a multi-entry irreducible cycle is
//! still detected — [`Cfg::loop_headers`] reports *every* entry of such a cycle.
//!
//! Data-oriented per the project rules: flat `Vec`s indexed by [`svm_ir::BlockIdx`], integer indices
//! throughout, no per-node heap nodes. Every traversal is **iterative** (explicit stacks) so a
//! pathologically deep CFG cannot overflow the host stack — the same fuzz-safety discipline the rest
//! of the pass holds.

use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Block, Terminator};

/// The **distinct** successor block indices of a terminator, in first-seen order. `Return` / tail
/// calls / `unreachable` have none. Deduplicated so CFG adjacency carries each edge once (a `br_if`
/// with equal arms, or a `br_table` repeating a target, contributes a single neighbor).
pub fn successors(term: &Terminator) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut push = |b: u32| {
        if !out.contains(&b) {
            out.push(b);
        }
    };
    match term {
        Terminator::Br { target, .. } => push(*target),
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => {
            push(*then_blk);
            push(*else_blk);
        }
        Terminator::BrTable {
            targets, default, ..
        } => {
            for (t, _) in targets {
                push(*t);
            }
            push(default.0);
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => {}
    }
    out
}

/// The control-flow graph of a function's blocks: successor and predecessor adjacency, plus the
/// derived traversals every pass leans on. Entry is always block `0` (§3b).
#[derive(Clone, Debug)]
pub struct Cfg {
    /// `succs[b]` = distinct successor blocks of `b` (from its terminator).
    pub succs: Vec<Vec<u32>>,
    /// `preds[b]` = distinct predecessor blocks of `b` (the inverse of `succs`).
    pub preds: Vec<Vec<u32>>,
}

impl Cfg {
    /// Build the graph from a function's blocks. Successor indices are taken as-is (a verified module
    /// guarantees they are in range); any stray out-of-range target is ignored rather than panicking,
    /// so this stays safe to run on unverified IR (fuzz discipline).
    pub fn new(blocks: &[Block]) -> Cfg {
        let n = blocks.len();
        let mut succs: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for b in blocks {
            let s: Vec<u32> = successors(&b.term)
                .into_iter()
                .filter(|&t| (t as usize) < n)
                .collect();
            succs.push(s);
        }
        for (b, ss) in succs.iter().enumerate() {
            for &s in ss {
                let p = &mut preds[s as usize];
                if !p.contains(&(b as u32)) {
                    p.push(b as u32);
                }
            }
        }
        Cfg { succs, preds }
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.succs.len()
    }
    /// Whether the function has no blocks (never true for a valid function, but total for callers).
    pub fn is_empty(&self) -> bool {
        self.succs.is_empty()
    }

    /// Depth-first **postorder** of the blocks reachable from entry (block `0`). Unreachable blocks
    /// are omitted. Iterative — no recursion.
    pub fn postorder(&self) -> Vec<u32> {
        let n = self.len();
        let mut order = Vec::with_capacity(n);
        if n == 0 {
            return order;
        }
        let mut visited = vec![false; n];
        // Explicit DFS stack of (block, index of the next successor to visit).
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        visited[0] = true;
        while let Some(&(node, idx)) = stack.last() {
            if idx < self.succs[node as usize].len() {
                stack.last_mut().unwrap().1 += 1;
                let s = self.succs[node as usize][idx];
                if !visited[s as usize] {
                    visited[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                order.push(node);
                stack.pop();
            }
        }
        order
    }

    /// **Reverse postorder** from entry — the canonical order for forward data-flow (a block appears
    /// before all its non-back-edge successors). Reachable blocks only.
    pub fn rpo(&self) -> Vec<u32> {
        let mut po = self.postorder();
        po.reverse();
        po
    }

    /// Immediate dominators, one entry per block: `Some(idom)` for a reachable non-entry block,
    /// `None` for the entry and for unreachable blocks. Iterative Cooper–Harvey–Kennedy over the RPO
    /// — **correct on irreducible CFGs**, no dominator-tree recursion.
    pub fn dominators(&self) -> Vec<Option<u32>> {
        let n = self.len();
        let mut idom = vec![u32::MAX; n]; // MAX = undefined
        if n == 0 {
            return Vec::new();
        }
        let rpo = self.rpo();
        let mut rpo_pos = vec![u32::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_pos[b as usize] = i as u32;
        }
        idom[0] = 0; // the entry is its own idom (sentinel for "processed")

        // Walk two nodes up the partially-built dominator tree to their common dominator.
        let intersect = |mut a: u32, mut b: u32, idom: &[u32]| -> u32 {
            while a != b {
                while rpo_pos[a as usize] > rpo_pos[b as usize] {
                    a = idom[a as usize];
                }
                while rpo_pos[b as usize] > rpo_pos[a as usize] {
                    b = idom[b as usize];
                }
            }
            a
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new_idom = u32::MAX;
                for &p in &self.preds[b as usize] {
                    if idom[p as usize] == u32::MAX {
                        continue; // predecessor not processed yet
                    }
                    new_idom = if new_idom == u32::MAX {
                        p
                    } else {
                        intersect(p, new_idom, &idom)
                    };
                }
                if new_idom != u32::MAX && idom[b as usize] != new_idom {
                    idom[b as usize] = new_idom;
                    changed = true;
                }
            }
        }

        idom.iter()
            .enumerate()
            .map(|(b, &d)| {
                if b == 0 || d == u32::MAX {
                    None // entry, or unreachable
                } else {
                    Some(d)
                }
            })
            .collect()
    }

    /// Strongly-connected components, via iterative Tarjan. Each component is a `Vec` of block
    /// indices; components come out in reverse topological order. A component is **cyclic** (part of
    /// a loop) iff it has more than one block, or one block with a self-edge. Irreducible-aware: a
    /// multi-entry cycle is a single component.
    pub fn sccs(&self) -> Vec<Vec<u32>> {
        let n = self.len();
        let mut index = vec![u32::MAX; n];
        let mut low = vec![0u32; n];
        let mut on_stack = vec![false; n];
        let mut comp_stack: Vec<u32> = Vec::new();
        let mut next = 0u32;
        let mut out: Vec<Vec<u32>> = Vec::new();

        for start in 0..n as u32 {
            if index[start as usize] != u32::MAX {
                continue;
            }
            // Iterative DFS: a call stack of (block, next-successor-index).
            let mut call: Vec<(u32, usize)> = vec![(start, 0)];
            index[start as usize] = next;
            low[start as usize] = next;
            next += 1;
            comp_stack.push(start);
            on_stack[start as usize] = true;

            while let Some(&(v, ci)) = call.last() {
                if ci < self.succs[v as usize].len() {
                    call.last_mut().unwrap().1 += 1;
                    let w = self.succs[v as usize][ci];
                    if index[w as usize] == u32::MAX {
                        index[w as usize] = next;
                        low[w as usize] = next;
                        next += 1;
                        comp_stack.push(w);
                        on_stack[w as usize] = true;
                        call.push((w, 0));
                    } else if on_stack[w as usize] && index[w as usize] < low[v as usize] {
                        low[v as usize] = index[w as usize];
                    }
                } else {
                    // v is fully explored; if it roots an SCC, pop the component.
                    if low[v as usize] == index[v as usize] {
                        let mut comp = Vec::new();
                        loop {
                            let x = comp_stack.pop().unwrap();
                            on_stack[x as usize] = false;
                            comp.push(x);
                            if x == v {
                                break;
                            }
                        }
                        out.push(comp);
                    }
                    call.pop();
                    if let Some(&(parent, _)) = call.last() {
                        if low[v as usize] < low[parent as usize] {
                            low[parent as usize] = low[v as usize];
                        }
                    }
                }
            }
        }
        out
    }

    /// A per-block flag: `true` iff the block is a **loop header** — a block inside a cyclic SCC that
    /// is entered from outside that SCC (or is the function entry). Irreducible-aware: an irreducible
    /// loop with two entries flags *both*. This is the primitive LICM / unrolling build on.
    pub fn loop_headers(&self) -> Vec<bool> {
        let n = self.len();
        let mut scc_of = vec![u32::MAX; n]; // SCC id, only for cyclic components
        for (id, comp) in self.sccs().iter().enumerate() {
            let cyclic = comp.len() > 1
                || (comp.len() == 1 && self.succs[comp[0] as usize].contains(&comp[0]));
            if cyclic {
                for &b in comp {
                    scc_of[b as usize] = id as u32;
                }
            }
        }
        let mut header = vec![false; n];
        for b in 0..n as u32 {
            let id = scc_of[b as usize];
            if id == u32::MAX {
                continue; // not in a loop
            }
            if b == 0 {
                header[0] = true; // the entry, if it sits in a cycle, is a header
                continue;
            }
            // Entered from outside the SCC ⇒ a header.
            if self.preds[b as usize]
                .iter()
                .any(|&p| scc_of[p as usize] != id)
            {
                header[b as usize] = true;
            }
        }
        header
    }
}
