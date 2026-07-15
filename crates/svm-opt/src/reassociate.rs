//! Constant reassociation (see `OPT.md` Phase 2): `(x OP c1) OP c2 → x OP (c1 OP c2)` for an
//! associative-and-commutative integer op, folding the two constants into one. It shrinks constant
//! chains (address math, index arithmetic) by an op each and exposes more folding/CSE downstream —
//! e.g. `(p + 4) + 4` and `(p + 8)` both become `p + 8` and then CSE together.
//!
//! Sound only for ops that are **both associative and commutative** mod 2^n — `Add`, `Mul`, `And`,
//! `Or`, `Xor` — so the regrouping and the constant combination are exact (`Sub` and the shifts are
//! excluded). The combined constant is computed with the shared [`fold_int_bin`] (interpreter-exact).
//!
//! Runs on the internal SSA form so the freshly-materialized constant is easy to introduce (prepended
//! to the block, before every use). The dead inner op is left for the cleanup fixpoint's DCE. Only
//! constants move; nothing crosses a block, so no parameter threading is needed.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use svm_ir::{BinOp, Func, Inst};

use crate::ssa::{from_ssa, to_ssa, SsaFunc, Value};
use crate::{const_value, fold_int_bin, Known};

/// Integer ops that are associative **and** commutative, so `(x OP c1) OP c2 == x OP (c1 OP c2)`.
fn is_reassoc(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor
    )
}

/// Negate an integer constant (`-c` mod 2^n), for normalizing `x - c` into `x + (-c)`.
fn negate(k: Known) -> Option<Known> {
    match k {
        Known::I32(v) => Some(Known::I32(v.wrapping_neg())),
        Known::I64(v) => Some(Known::I64(v.wrapping_neg())),
        _ => None,
    }
}

/// Intern a constant `k` into the block's pending new-constant list, returning its (stable) value id.
fn intern(
    kmap: &mut BTreeMap<Known, Value>,
    new_consts: &mut Vec<Known>,
    base: Value,
    k: Known,
) -> Value {
    match kmap.get(&k) {
        Some(&id) => id,
        None => {
            let id = base + new_consts.len() as u32;
            kmap.insert(k, id);
            new_consts.push(k);
            id
        }
    }
}

/// Reassociate constant chains in every block of a function.
pub fn reassociate(f: &Func, fn_results: &[usize]) -> Func {
    let mut s = to_ssa(f, fn_results);
    for bi in 0..s.blocks.len() {
        // Each pass collapses one level; iterate so `((x+a)+b)+c` fully folds. Cap guards pathologies.
        for _ in 0..16 {
            if !reassociate_block(&mut s, bi, fn_results) {
                break;
            }
        }
    }
    from_ssa(&s)
}

/// One reassociation pass over block `bi`. Returns whether it rewrote anything.
fn reassociate_block(s: &mut SsaFunc, bi: usize, fn_results: &[usize]) -> bool {
    let nparams = s.blocks[bi].params.len();

    // Value → its constant (if a literal `const`) and Value → its defining instruction.
    let mut cst: BTreeMap<Value, Known> = BTreeMap::new();
    let mut def: BTreeMap<Value, Inst> = BTreeMap::new();
    let mut slot = nparams;
    for inst in &s.blocks[bi].insts {
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            let v = s.values[bi][slot];
            if let Some(k) = const_value(inst) {
                cst.insert(v, k);
            }
            def.insert(v, inst.clone());
        }
        slot += rc;
    }

    // Find rewrites (immutable scan). New constants get ids `base + position`; `num_values` is bumped
    // only when we commit, so nothing mutates `s` during the scan.
    let base = s.num_values;
    let mut kmap: BTreeMap<Known, Value> = BTreeMap::new();
    let mut new_consts: Vec<Known> = Vec::new();
    // outer value → (new op, variable operand, combined-const value).
    let mut rewrites: BTreeMap<Value, (BinOp, Value, Value)> = BTreeMap::new();
    let mut slot = nparams;
    for inst in &s.blocks[bi].insts {
        let rc = inst.result_count(fn_results);
        if rc == 1 {
            let ov = s.values[bi][slot];
            if let &Inst::IntBin { ty, op, a, b } = inst {
                if op == BinOp::Sub {
                    // Normalize `x - c` → `x + (-c)` so subtraction-by-constant chains reassociate
                    // through `Add` too (e.g. `p + 16 - 4` collapses).
                    if let Some(neg) = cst.get(&b).copied().and_then(negate) {
                        let kv = intern(&mut kmap, &mut new_consts, base, neg);
                        rewrites.insert(ov, (BinOp::Add, a, kv));
                    }
                } else if is_reassoc(op) {
                    // The outer op: one constant operand `c`, one variable `var`.
                    let outer = cst
                        .get(&b)
                        .map(|&kc| (a, kc))
                        .or_else(|| cst.get(&a).map(|&kc| (b, kc)));
                    if let Some((var, c)) = outer {
                        // The variable operand must itself be `<same ty/op>(inner_var, inner_const)`.
                        if let Some(&Inst::IntBin {
                            ty: ty2,
                            op: op2,
                            a: ia,
                            b: ib,
                        }) = def.get(&var)
                        {
                            if ty2 == ty && op2 == op {
                                let inner = cst
                                    .get(&ib)
                                    .map(|&kc| (ia, kc))
                                    .or_else(|| cst.get(&ia).map(|&kc| (ib, kc)));
                                if let Some((ivar, ic)) = inner {
                                    if let Some(k) = fold_int_bin(ty, op, ic, c) {
                                        let kv = intern(&mut kmap, &mut new_consts, base, k);
                                        rewrites.insert(ov, (op, ivar, kv));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        slot += rc;
    }

    if rewrites.is_empty() {
        return false;
    }

    // Commit: prepend the new constants, rewrite the outer ops to `var OP combined_const`. The block
    // layout stays consistent for `from_ssa`: values = params ++ new-consts ++ original results, and
    // insts = new-const insts ++ original insts (rewritten).
    s.num_values = base + new_consts.len() as u32;
    let mut new_insts: Vec<Inst> = new_consts.iter().map(|k| k.to_const_inst()).collect();
    let mut new_values: Vec<Value> = s.values[bi][..nparams].to_vec();
    for i in 0..new_consts.len() {
        new_values.push(base + i as u32);
    }

    let orig = core::mem::take(&mut s.blocks[bi].insts);
    let mut slot = nparams;
    for inst in orig {
        let rc = inst.result_count(fn_results);
        let ni = if rc == 1 {
            let ov = s.values[bi][slot];
            match rewrites.get(&ov) {
                Some(&(new_op, ivar, kv)) => {
                    if let Inst::IntBin { ty, .. } = inst {
                        Inst::IntBin {
                            ty,
                            op: new_op,
                            a: ivar,
                            b: kv,
                        }
                    } else {
                        inst
                    }
                }
                None => inst,
            }
        } else {
            inst
        };
        new_insts.push(ni);
        slot += rc;
    }
    new_values.extend_from_slice(&s.values[bi][nparams..]);

    s.blocks[bi].insts = new_insts;
    s.values[bi] = new_values;
    true
}
