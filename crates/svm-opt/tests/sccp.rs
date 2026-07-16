//! Differential + structural spec for SCCP (OPT.md Phase 2). Each case is built so the constant is
//! only discoverable *across* blocks — a join value constant on multiple predecessors, a branch
//! whose selector is a cross-block-constant parameter, a loop-invariant constant — i.e. exactly what
//! the per-block passes cannot fold. We assert (1) `optimize_module` preserves behavior on the
//! reference interpreter (results + traps) and (2) SCCP actually fired (the arithmetic/branch it
//! targets is gone).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        debug_info: None,
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

/// Verify original + optimized, then assert byte-identical behavior over all arg sets. Returns the
/// optimized module for structural follow-up assertions.
fn check_equiv(orig: &Module, argsets: &[Vec<Value>]) -> Module {
    verify_module(orig).expect("original verifies");
    let opt = optimize_module(orig);
    verify_module(&opt).expect("optimized re-verifies");
    for args in argsets {
        assert_eq!(
            run(orig, args),
            run(&opt, args),
            "behavioral divergence on args {args:?}"
        );
    }
    opt
}

fn count<F: Fn(&Inst) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .filter(|i| pred(i))
        .count()
}

fn add(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Add,
        a,
        b,
    }
}
fn brif(cond: u32, t: u32, ta: Vec<u32>, e: u32, ea: Vec<u32>) -> Terminator {
    Terminator::BrIf {
        cond,
        then_blk: t,
        then_args: ta,
        else_blk: e,
        else_args: ea,
    }
}

#[test]
fn constant_through_multi_pred_block_param() {
    // b0(a): both br_if arms pass const 5 to b1 → b1's param is 5 regardless of `a`. The block has
    // two predecessors, so merge/fold can't reach it; only SCCP's phi meet can.
    //   b1(x): return x + 10   → always 15.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],     // v0 = a
                insts: vec![Inst::ConstI32(5)], // v1
                term: brif(0, 1, vec![1], 1, vec![1]),
            },
            Block {
                params: vec![ValType::I32],                 // v0 = x (= 5 on every edge)
                insts: vec![Inst::ConstI32(10), add(0, 1)], // v1=10, v2 = x + 10
                term: Terminator::Return(vec![2]),
            },
        ],
    };
    let m = module(f);
    assert_eq!(count(&m, |i| matches!(i, Inst::IntBin { .. })), 1);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(0)],
            vec![Value::I32(1)],
            vec![Value::I32(-9)],
        ],
    );
    // SCCP proved x == 5, so x + 10 folded to 15 — no arithmetic survives.
    assert_eq!(count(&opt, |i| matches!(i, Inst::IntBin { .. })), 0);
    assert_eq!(run(&opt, &[Value::I32(42)]), Ok(vec![Value::I32(15)]));
}

#[test]
fn conditional_branch_resolved_by_cross_block_constant() {
    // b0(a): both arms pass const 1 to b3 → b3's selector param is 1 → the branch always takes b4.
    // b3 has two preds (can't merge), so only SCCP can resolve the branch and kill b5.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],            // a
                insts: vec![Inst::ConstI32(1)],        // v1 = 1
                term: brif(0, 1, vec![1], 1, vec![1]), // both arms → b1 with 1
            },
            Block {
                params: vec![ValType::I32], // c (= 1)
                insts: vec![],
                term: brif(0, 2, vec![], 3, vec![]), // if c { b2 } else { b3 }
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(100)],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(200)], // dead once SCCP resolves the branch
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = module(f);
    assert_eq!(m.funcs[0].blocks.len(), 4);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(0)],
            vec![Value::I32(1)],
            vec![Value::I32(7)],
        ],
    );
    // b3 (the const 200 arm) is unreachable after SCCP → pruned; result is always 100.
    assert!(opt.funcs[0].blocks.len() < 4, "dead arm should be pruned");
    assert!(
        count(&opt, |i| matches!(i, Inst::ConstI32(200))) == 0,
        "the dead const 200 should be gone"
    );
    assert_eq!(run(&opt, &[Value::I32(3)]), Ok(vec![Value::I32(100)]));
}

#[test]
fn loop_invariant_constant_folds_inside_the_loop() {
    // b0(n): enter the loop with k = 7 (invariant) and i = n.
    // b1(k, i): d = k + k;  i2 = i - 1;  if i2 { loop back with (k, i2) } else { exit with d }.
    // b2(r): return r.  SCCP proves k == 7 around the back edge, so d = k + k folds to 14 — a fold
    // the per-block pass can't make (k is a parameter, unknown locally).
    let sub = |a: u32, b: u32| Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Sub,
        a,
        b,
    };
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],     // n
                insts: vec![Inst::ConstI32(7)], // v1 = 7
                term: Terminator::Br {
                    target: 1,
                    args: vec![1, 0], // (k=7, i=n)
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // v0=k, v1=i
                insts: vec![
                    add(0, 0),         // v2 = k + k     (SCCP: 14)
                    Inst::ConstI32(1), // v3 = 1
                    sub(1, 3),         // v4 = i - 1
                ],
                term: brif(4, 1, vec![0, 4], 2, vec![2]), // loop (k,i-1) else exit with d
            },
            Block {
                params: vec![ValType::I32], // r
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = module(f);
    // n = 0 loops forever (i-1 never hits 0) → both original and optimized exhaust fuel identically;
    // n >= 1 terminates returning 14.
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(1)],
            vec![Value::I32(2)],
            vec![Value::I32(5)],
            vec![Value::I32(0)],
        ],
    );
    // The loop-invariant `k + k` folded to the constant 14, so only the loop-decrement op survives
    // (`i - 1`, which reassociation normalizes to `i + (-1)`). Two integer ops → one.
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::IntBin { .. })),
        1,
        "loop-invariant k + k should fold, leaving only the loop decrement"
    );
    assert_eq!(run(&opt, &[Value::I32(3)]), Ok(vec![Value::I32(14)]));
}

#[test]
fn trapping_op_is_never_marked_constant() {
    // A cross-block-constant divisor of 0: SCCP must NOT fold `x / 0` (it would trap), so the
    // residual still traps identically. b0 passes const 0 to b1 on both arms.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],                          // a
                insts: vec![Inst::ConstI32(0), Inst::ConstI32(100)], // v1=0, v2=100
                term: brif(0, 1, vec![2, 1], 1, vec![2, 1]),         // pass (100, 0)
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // v0=num(100), v1=den(0)
                insts: vec![Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::DivS,
                    a: 0,
                    b: 1,
                }], // v2 = 100 / 0  → traps
                term: Terminator::Return(vec![2]),
            },
        ],
    };
    let m = module(f);
    let opt = check_equiv(&m, &[vec![Value::I32(0)], vec![Value::I32(1)]]);
    // The div must survive (folding it would have changed a trap into a constant).
    assert_eq!(
        count(&opt, |i| matches!(
            i,
            Inst::IntBin {
                op: BinOp::DivS,
                ..
            }
        )),
        1,
        "a trapping div-by-zero must not be folded away"
    );
    assert!(run(&opt, &[Value::I32(0)]).is_err(), "still traps");
}
