//! Differential spec for the Stage-0 optimizer: every optimized module must (1) re-verify
//! and (2) produce byte-identical results *and traps* to the original on the reference
//! interpreter, for every argument set. Plus structural checks that the intended rewrite
//! actually happened (folds collapsed, dead blocks vanished, trapping ops preserved).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, CmpOp, Func, Inst, IntTy, Module, Terminator, ValType};
use svm_peval::optimize_module;
use svm_verify::verify_module;

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

/// Verify original + optimized, then assert behavioral equivalence over all arg sets.
fn check_equiv(orig: &Module, argsets: &[Vec<Value>]) -> Module {
    verify_module(orig).expect("original module verifies");
    let opt = optimize_module(orig);
    verify_module(&opt).expect("optimized module re-verifies");
    for args in argsets {
        assert_eq!(
            run(orig, args),
            run(&opt, args),
            "behavioral divergence on args {args:?}"
        );
    }
    opt
}

fn single_block_fn(
    params: Vec<ValType>,
    results: Vec<ValType>,
    insts: Vec<Inst>,
    ret: u32,
) -> Module {
    Module {
        funcs: vec![Func {
            params: params.clone(),
            results,
            blocks: vec![Block {
                params,
                insts,
                term: Terminator::Return(vec![ret]),
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn folds_integer_arithmetic_chain() {
    // () -> i64 : (2 + 3) * 4  ==  20
    let m = single_block_fn(
        vec![],
        vec![ValType::I64],
        vec![
            Inst::ConstI64(2), // 0
            Inst::ConstI64(3), // 1
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 1,
            }, // 2 = 5
            Inst::ConstI64(4), // 3
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Mul,
                a: 2,
                b: 3,
            }, // 4 = 20
        ],
        4,
    );

    let opt = check_equiv(&m, &[vec![]]);
    // Both IntBins collapsed to constants in place (indices preserved).
    assert!(matches!(opt.funcs[0].blocks[0].insts[2], Inst::ConstI64(5)));
    assert!(matches!(
        opt.funcs[0].blocks[0].insts[4],
        Inst::ConstI64(20)
    ));
    assert_eq!(run(&opt, &[]), Ok(vec![Value::I64(20)]));
}

#[test]
fn folds_compare_and_conversion() {
    // () -> i64 : extend_u( (7 <u 9) )  == 1
    let m = single_block_fn(
        vec![],
        vec![ValType::I64],
        vec![
            Inst::ConstI32(7), // 0
            Inst::ConstI32(9), // 1
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::LtU,
                a: 0,
                b: 1,
            }, // 2 = 1 (i32)
            Inst::Convert {
                op: svm_ir::ConvOp::ExtendI32U,
                a: 2,
            }, // 3 = 1 (i64)
        ],
        3,
    );
    let opt = check_equiv(&m, &[vec![]]);
    assert!(matches!(opt.funcs[0].blocks[0].insts[2], Inst::ConstI32(1)));
    assert!(matches!(opt.funcs[0].blocks[0].insts[3], Inst::ConstI64(1)));
}

#[test]
fn resolves_branch_and_prunes_dead_block() {
    // (x: i64) -> i64 : if 1 { x + 10 } else { x + 20 }  -> always x + 10
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64],     // x = 0
                    insts: vec![Inst::ConstI32(1)], // 1
                    term: Terminator::BrIf {
                        cond: 1,
                        then_blk: 1,
                        then_args: vec![0],
                        else_blk: 2,
                        else_args: vec![0],
                    },
                },
                Block {
                    params: vec![ValType::I64], // y = 0
                    insts: vec![
                        Inst::ConstI64(10), // 1
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 0,
                            b: 1,
                        }, // 2
                    ],
                    term: Terminator::Return(vec![2]),
                },
                Block {
                    params: vec![ValType::I64],
                    insts: vec![
                        Inst::ConstI64(20), // 1
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 0,
                            b: 1,
                        }, // 2
                    ],
                    term: Terminator::Return(vec![2]),
                },
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![Value::I64(5)], vec![Value::I64(-100)]]);
    // The else-block is gone; the conditional became an unconditional branch.
    assert_eq!(opt.funcs[0].blocks.len(), 2);
    assert!(matches!(
        opt.funcs[0].blocks[0].term,
        Terminator::Br { target: 1, .. }
    ));
    assert_eq!(run(&opt, &[Value::I64(5)]), Ok(vec![Value::I64(15)]));
}

#[test]
fn resolves_br_table_and_prunes() {
    // () -> i32 : br_table over const 1 -> selects targets[1] (block 2, returns 200)
    let blk = |c: i32| Block {
        params: vec![],
        insts: vec![Inst::ConstI32(c)],
        term: Terminator::Return(vec![0]),
    };
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![
                Block {
                    params: vec![],
                    insts: vec![Inst::ConstI32(1)], // 0
                    term: Terminator::BrTable {
                        idx: 0,
                        targets: vec![(1, vec![]), (2, vec![])],
                        default: (3, vec![]),
                    },
                },
                blk(100),
                blk(200),
                blk(300),
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![]]);
    assert_eq!(opt.funcs[0].blocks.len(), 2); // entry + selected target only
    assert_eq!(run(&opt, &[]), Ok(vec![Value::I32(200)]));
}

#[test]
fn does_not_fold_trapping_div_and_preserves_trap() {
    // () -> i32 : 10 / 0  (div_s)  must NOT fold; both must trap DivByZero.
    let m = single_block_fn(
        vec![],
        vec![ValType::I32],
        vec![
            Inst::ConstI32(10), // 0
            Inst::ConstI32(0),  // 1
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::DivS,
                a: 0,
                b: 1,
            }, // 2
        ],
        2,
    );
    let opt = check_equiv(&m, &[vec![]]);
    // The division survives the optimizer untouched.
    assert!(matches!(
        opt.funcs[0].blocks[0].insts[2],
        Inst::IntBin {
            op: BinOp::DivS,
            ..
        }
    ));
    assert_eq!(run(&opt, &[]), Err(Trap::DivByZero));
}

#[test]
fn signed_min_div_neg_one_not_folded() {
    // () -> i32 : i32::MIN / -1  must NOT fold; both must trap IntOverflow.
    let m = single_block_fn(
        vec![],
        vec![ValType::I32],
        vec![
            Inst::ConstI32(i32::MIN), // 0
            Inst::ConstI32(-1),       // 1
            Inst::IntBin {
                ty: IntTy::I32,
                op: BinOp::DivS,
                a: 0,
                b: 1,
            }, // 2
        ],
        2,
    );
    let opt = check_equiv(&m, &[vec![]]);
    assert!(matches!(
        opt.funcs[0].blocks[0].insts[2],
        Inst::IntBin {
            op: BinOp::DivS,
            ..
        }
    ));
    assert_eq!(run(&opt, &[]), Err(Trap::IntOverflow));
}

#[test]
fn preserves_loops_with_nonconstant_conditions() {
    // (n: i32) -> i32 : sum_{k=1..=n} k, a real back-edge loop. The header compare is
    // data-dependent, so nothing folds away and no block is pruned — the optimizer must
    // leave the loop structurally intact and behaviorally identical.
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I32],
            results: vec![ValType::I32],
            blocks: vec![
                // block 0: entry(n) -> header(acc=0, i=n)
                Block {
                    params: vec![ValType::I32],     // n = 0
                    insts: vec![Inst::ConstI32(0)], // 1
                    term: Terminator::Br {
                        target: 1,
                        args: vec![1, 0],
                    },
                },
                // block 1: header(acc, i): if i == 0 -> exit(acc) else body(acc, i)
                Block {
                    params: vec![ValType::I32, ValType::I32], // acc=0, i=1
                    insts: vec![
                        Inst::ConstI32(0), // 2
                        Inst::IntCmp {
                            ty: IntTy::I32,
                            op: CmpOp::Eq,
                            a: 1,
                            b: 2,
                        }, // 3
                    ],
                    term: Terminator::BrIf {
                        cond: 3,
                        then_blk: 2,
                        then_args: vec![0],
                        else_blk: 3,
                        else_args: vec![0, 1],
                    },
                },
                // block 2: exit(acc) -> return acc
                Block {
                    params: vec![ValType::I32],
                    insts: vec![],
                    term: Terminator::Return(vec![0]),
                },
                // block 3: body(acc, i): acc+=i; i-=1; -> header
                Block {
                    params: vec![ValType::I32, ValType::I32], // acc=0, i=1
                    insts: vec![
                        Inst::IntBin {
                            ty: IntTy::I32,
                            op: BinOp::Add,
                            a: 0,
                            b: 1,
                        }, // 2 = acc+i
                        Inst::ConstI32(1), // 3
                        Inst::IntBin {
                            ty: IntTy::I32,
                            op: BinOp::Sub,
                            a: 1,
                            b: 3,
                        }, // 4 = i-1
                    ],
                    term: Terminator::Br {
                        target: 1,
                        args: vec![2, 4],
                    },
                },
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(0)],
            vec![Value::I32(1)],
            vec![Value::I32(5)],
            vec![Value::I32(10)],
        ],
    );
    // All four blocks survive (nothing constant-foldable controls the flow).
    assert_eq!(opt.funcs[0].blocks.len(), 4);
    assert_eq!(run(&opt, &[Value::I32(5)]), Ok(vec![Value::I32(15)]));
}
