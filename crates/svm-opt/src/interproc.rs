//! Interprocedural passes (see `OPT.md` Phase 3). These are **module-level** — they add, remove, or
//! renumber whole functions — unlike the per-function scalar passes, so they run once over the module
//! rather than inside `optimize_func`. Output is re-verified like everything else in this crate, so a
//! bug here is a clean verify error, never an escape (untrusted-for-escape posture, §20a).
//!
//! Slice 1 is **dead-function elimination**: drop functions that no reachable code can call. Direct-
//! call inlining and constant-funcref devirtualization build on the same call-graph plumbing and land
//! next.

use alloc::vec;
use alloc::vec::Vec;
use svm_ir::{Export, Func, FuncIdx, Inst, Module, Terminator};

/// Visit every **static function index** a function references: a direct `call`, a `ref.func`, a
/// `thread.spawn` entry, and the `return_call` terminator. This mirrors `svm_ir::offset_func_indices`
/// (the merged-module reindexer) exactly — `call_indirect` / `cont.*` dispatch on runtime funcref
/// *values* (an `i32` that equals the function index; the identity table), and `call.import` carries
/// an import index, so none of those name a *static* callee. A `ref.func` **does**: it materializes a
/// callable funcref, so a function whose reference is taken by reachable code must be kept — it may be
/// reached by a later `call_indirect`. Counting `ref.func` as a call edge is the sound over-
/// approximation that keeps such functions live.
fn referenced_funcs(f: &Func, mut visit: impl FnMut(FuncIdx)) {
    for b in &f.blocks {
        for inst in &b.insts {
            match inst {
                Inst::Call { func, .. }
                | Inst::RefFunc { func }
                | Inst::ThreadSpawn { func, .. } => visit(*func),
                _ => {}
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            visit(*func);
        }
    }
}

/// Whether the module dispatches on a runtime **funcref value** — `call_indirect`,
/// `return_call_indirect`, or `cont.new` (whose `func` operand is a funcref value, not a static
/// index). Because a funcref equals its function index (the identity table) and can be a plain
/// `ConstI32` indistinguishable from ordinary data, an indirect dispatch could reach *any* in-range
/// function, so removing or renumbering functions is unsound without funcref value-flow analysis.
/// Slice-1 dead-function elimination therefore leaves such modules untouched; the later
/// devirtualization pass folds constant indirect calls to direct ones, after which this pass applies.
/// (`thread.spawn` carries a *static* funcidx — [`referenced_funcs`] already handles it — so it is not
/// a value dispatch and does not gate here.)
fn has_indirect_funcref_dispatch(m: &Module) -> bool {
    m.funcs.iter().flat_map(|f| &f.blocks).any(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i, Inst::CallIndirect { .. } | Inst::ContNew { .. }))
            || matches!(b.term, Terminator::ReturnCallIndirect { .. })
    })
}

/// Rewrite every static function index in `f` through the old→new map (the exact set
/// [`referenced_funcs`] reads).
fn remap_func_indices(f: &mut Func, map: &[u32]) {
    for b in &mut f.blocks {
        for inst in &mut b.insts {
            match inst {
                Inst::Call { func, .. }
                | Inst::RefFunc { func }
                | Inst::ThreadSpawn { func, .. } => *func = map[*func as usize],
                _ => {}
            }
        }
        if let Terminator::ReturnCall { func, .. } = &mut b.term {
            *func = map[*func as usize];
        }
    }
}

/// **Dead-function elimination.** Drop every function unreachable in the call graph from the roots —
/// the conventional entry (`func 0`, what `run(m, 0, …)` invokes) and every named export — following
/// `call` / `return_call` / `thread.spawn` edges and, conservatively, `ref.func` (a materialized
/// funcref can be reached by a later `call_indirect`). Surviving functions are renumbered densely and
/// every static funcidx reference (and each export) is remapped; `call_indirect` is untouched because
/// it dispatches on the funcref value, which equals the function index and so rides the same map only
/// where a `ref.func` produced it (already remapped above).
///
/// Sound: a dropped function is provably uncallable from any reachable code, so removing it changes no
/// observable behavior. Conservative on `ref.func` (a reference taken but never indirectly called
/// still keeps its function) — correct, never over-eager. Debug info is dropped when any function is
/// removed (its `(func, …)` positions would be stale after renumbering; it is strippable and untrusted
/// for escape, §3a).
pub fn dead_func_elim(m: &Module) -> Module {
    let n = m.funcs.len();
    if n == 0 {
        return m.clone();
    }
    // Unsound to remove/renumber functions while any indirect funcref dispatch is live (see
    // [`has_indirect_funcref_dispatch`]) — bail to the identity until devirtualization removes it.
    if has_indirect_funcref_dispatch(m) {
        return m.clone();
    }

    // Reachability closure from the roots.
    let mut reachable = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mark = |i: usize, reachable: &mut [bool], stack: &mut Vec<usize>| {
        if i < n && !reachable[i] {
            reachable[i] = true;
            stack.push(i);
        }
    };
    mark(0, &mut reachable, &mut stack); // conventional entry
    for e in &m.exports {
        mark(e.func as usize, &mut reachable, &mut stack);
    }
    while let Some(fi) = stack.pop() {
        referenced_funcs(&m.funcs[fi], |g| {
            let g = g as usize;
            if g < n && !reachable[g] {
                reachable[g] = true;
                stack.push(g);
            }
        });
    }

    if reachable.iter().all(|&r| r) {
        return m.clone(); // nothing dead — identity (no renumbering)
    }

    // old funcidx → new (dense over the survivors, order-preserving).
    let mut map = vec![0u32; n];
    let mut next = 0u32;
    for (i, &live) in reachable.iter().enumerate() {
        if live {
            map[i] = next;
            next += 1;
        }
    }

    let funcs: Vec<Func> = (0..n)
        .filter(|&i| reachable[i])
        .map(|i| {
            let mut f = m.funcs[i].clone();
            remap_func_indices(&mut f, &map);
            f
        })
        .collect();
    // Every export is a root, hence reachable; remap its funcidx through the survivor map.
    let exports: Vec<Export> = m
        .exports
        .iter()
        .map(|e| Export {
            name: e.name.clone(),
            func: map[e.func as usize],
        })
        .collect();

    Module {
        funcs,
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports,
        debug_info: None, // positions go stale once functions are renumbered
    }
}
