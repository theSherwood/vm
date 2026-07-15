//! Differential spec for the Stage-0 optimizer: every optimized module must (1) re-verify
//! and (2) produce byte-identical results *and traps* to the original on the reference
//! interpreter, for every argument set. Plus structural checks that the intended rewrite
//! actually happened (folds collapsed, dead blocks vanished, trapping ops preserved).

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Func, Inst, IntTy, IntUnOp, LoadOp, Memory, Module, StoreOp, Terminator,
    ValType,
};
use svm_opt::optimize_module;
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
    // Fold + DCE collapse the whole chain to a single constant (the dead intermediates go).
    let insts = &opt.funcs[0].blocks[0].insts;
    assert_eq!(insts.len(), 1);
    assert!(matches!(insts[0], Inst::ConstI64(20)));
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
    // Compare + convert fold, then DCE drops the now-dead i32 sources, leaving one i64 const.
    let insts = &opt.funcs[0].blocks[0].insts;
    assert_eq!(insts.len(), 1);
    assert!(matches!(insts[0], Inst::ConstI64(1)));
    assert_eq!(run(&opt, &[]), Ok(vec![Value::I64(1)]));
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
    // The branch resolves, the else-block is pruned, and the lone taken successor merges back
    // into the entry — leaving a single block that returns x + 10.
    assert_eq!(opt.funcs[0].blocks.len(), 1);
    assert!(matches!(opt.funcs[0].blocks[0].term, Terminator::Return(_)));
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
    // Table resolves to one edge, the unselected blocks are pruned, and the lone target merges
    // into the entry — a single block returning 200.
    assert_eq!(opt.funcs[0].blocks.len(), 1);
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

// ----- CFG cleanup: block merging + dead-parameter elimination -----

#[test]
fn merges_straight_line_chain() {
    // (x: i64) -> i64 : entry -> b1 -> b2, all unconditional, computing x + 1.
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64], // x
                    insts: vec![],
                    term: Terminator::Br {
                        target: 1,
                        args: vec![0],
                    },
                },
                Block {
                    params: vec![ValType::I64], // a
                    insts: vec![
                        Inst::ConstI64(1), // 1
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 0,
                            b: 1,
                        }, // 2 = a + 1
                    ],
                    term: Terminator::Br {
                        target: 2,
                        args: vec![2],
                    },
                },
                Block {
                    params: vec![ValType::I64], // b
                    insts: vec![],
                    term: Terminator::Return(vec![0]),
                },
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![Value::I64(10)], vec![Value::I64(-1)]]);
    // The whole chain fuses into one block.
    assert_eq!(opt.funcs[0].blocks.len(), 1);
    assert_eq!(run(&opt, &[Value::I64(10)]), Ok(vec![Value::I64(11)]));
}

#[test]
fn drops_dead_block_parameter_across_predecessors() {
    // A diamond whose join block has two predecessors (so it cannot be merged) and an unused
    // `junk` parameter that both edges supply. The parameter and both edge args must be dropped.
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![
                        Inst::ConstI64(0), // 1
                        Inst::IntCmp {
                            ty: IntTy::I64,
                            op: CmpOp::Ne,
                            a: 0,
                            b: 1,
                        }, // 2: x != 0
                        Inst::ConstI64(111), // 3: junkA
                        Inst::ConstI64(222), // 4: junkB
                    ],
                    term: Terminator::BrIf {
                        cond: 2,
                        then_blk: 1,
                        then_args: vec![0, 3], // join(x, junkA)
                        else_blk: 1,
                        else_args: vec![1, 4], // join(0, junkB)
                    },
                },
                Block {
                    params: vec![ValType::I64, ValType::I64], // 0: v, 1: junk (unused)
                    insts: vec![],
                    term: Terminator::Return(vec![0]),
                },
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(
        &m,
        &[
            vec![Value::I64(5)],
            vec![Value::I64(0)],
            vec![Value::I64(-9)],
        ],
    );
    // The join keeps its two predecessors (not merged), but the dead `junk` parameter is gone,
    // and with it the now-dead 111/222 constants.
    assert!(opt.funcs[0].blocks.iter().all(|b| b.params.len() <= 1));
    assert!(!opt.funcs[0]
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, Inst::ConstI64(111) | Inst::ConstI64(222))));
    assert!(opt.funcs[0]
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrIf { .. })));
    assert_eq!(run(&opt, &[Value::I64(5)]), Ok(vec![Value::I64(5)]));
}

// ----- Stage 0.x: dead-value elimination -----

#[test]
fn drops_dead_arithmetic_keeps_live_path() {
    // (x: i64) -> i64 : a dead 100+200 chain alongside the live x+7.
    let m = single_block_fn(
        vec![ValType::I64],
        vec![ValType::I64],
        vec![
            Inst::ConstI64(100), // 1
            Inst::ConstI64(200), // 2
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1,
                b: 2,
            }, // 3 = 300 (dead)
            Inst::ConstI64(7),   // 4
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 4,
            }, // 5 = x + 7 (live)
        ],
        5,
    );

    let opt = check_equiv(&m, &[vec![Value::I64(10)], vec![Value::I64(-1)]]);
    let insts = &opt.funcs[0].blocks[0].insts;
    // Only the live const 7 and the live add survive; the dead 100/200/300 chain is gone.
    assert_eq!(insts.len(), 2);
    assert!(matches!(insts[0], Inst::ConstI64(7)));
    assert!(matches!(insts[1], Inst::IntBin { op: BinOp::Add, .. }));
    assert_eq!(run(&opt, &[Value::I64(10)]), Ok(vec![Value::I64(17)]));
}

#[test]
fn keeps_dead_load_but_drops_dead_arithmetic() {
    // A load can fault, so a *dead* load must be kept; a dead pure add is removed.
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(0), // 0 : addr
                    Inst::Load {
                        op: LoadOp::I32,
                        addr: 0,
                        offset: 0,
                        align: 0,
                    }, // 1 : dead result, but kept (can fault)
                    Inst::ConstI32(3), // 2
                    Inst::ConstI32(4), // 3
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 2,
                        b: 3,
                    }, // 4 = 7 (dead)
                    Inst::ConstI32(7), // 5
                ],
                term: Terminator::Return(vec![5]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![]]);
    let insts = &opt.funcs[0].blocks[0].insts;
    assert_eq!(insts.len(), 3); // addr const, the load, and the returned const 7
    assert_eq!(
        insts
            .iter()
            .filter(|i| matches!(i, Inst::Load { .. }))
            .count(),
        1,
        "the dead load must be preserved"
    );
    assert!(
        !insts.iter().any(|i| matches!(i, Inst::IntBin { .. })),
        "the dead add must be removed"
    );
    assert_eq!(run(&opt, &[]), Ok(vec![Value::I32(7)]));
}

#[test]
fn keeps_store_effect_across_renumbering() {
    // A store has a side effect (and produces no SSA result), so it is kept even with a dead
    // pure op before it — and the zero-result store must not corrupt value renumbering.
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(8),  // 0 : addr
                    Inst::ConstI32(42), // 1 : value
                    Inst::ConstI32(5),  // 2 (dead)
                    Inst::ConstI32(6),  // 3 (dead)
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Mul,
                        a: 2,
                        b: 3,
                    }, // 4 = 30 (dead)
                    Inst::Store {
                        op: StoreOp::I32,
                        addr: 0,
                        value: 1,
                        offset: 0,
                        align: 0,
                    }, // (no result)
                    Inst::ConstI32(0),  // 5
                ],
                term: Terminator::Return(vec![5]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![]]);
    let insts = &opt.funcs[0].blocks[0].insts;
    assert_eq!(
        insts
            .iter()
            .filter(|i| matches!(i, Inst::Store { .. }))
            .count(),
        1,
        "the store must be preserved"
    );
    assert!(!insts.iter().any(|i| matches!(i, Inst::IntBin { .. })));
    assert_eq!(run(&opt, &[]), Ok(vec![Value::I32(0)]));
}

#[test]
fn dce_removes_dead_branch_condition() {
    // After the constant `br_if` resolves to `br`, the whole condition computation is dead.
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64], // x = 0
                    insts: vec![
                        Inst::ConstI32(5), // 1
                        Inst::ConstI32(5), // 2
                        Inst::IntCmp {
                            ty: IntTy::I32,
                            op: CmpOp::Eq,
                            a: 1,
                            b: 2,
                        }, // 3 = 1
                    ],
                    term: Terminator::BrIf {
                        cond: 3,
                        then_blk: 1,
                        then_args: vec![0],
                        else_blk: 2,
                        else_args: vec![0],
                    },
                },
                Block {
                    params: vec![ValType::I64],
                    insts: vec![],
                    term: Terminator::Return(vec![0]),
                },
                Block {
                    params: vec![ValType::I64],
                    insts: vec![Inst::ConstI64(999)],
                    term: Terminator::Return(vec![1]),
                },
            ],
        }],
        ..Default::default()
    };

    let opt = check_equiv(&m, &[vec![Value::I64(5)]]);
    // The condition computation is dead and gone; the resolved branch's lone successor merges
    // into the entry, which now just returns x.
    assert_eq!(opt.funcs[0].blocks.len(), 1);
    assert!(
        opt.funcs[0].blocks[0].insts.is_empty(),
        "dead condition computation should be gone"
    );
    assert!(matches!(opt.funcs[0].blocks[0].term, Terminator::Return(_)));
    assert_eq!(run(&opt, &[Value::I64(5)]), Ok(vec![Value::I64(5)]));
}

/// A tiny deterministic LCG so the fuzz is reproducible without a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33 ^ self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn fuzz_arithmetic_dag_equivalence_and_shrinks() {
    // Build random all-`i64` straight-line DAGs of pure, non-trapping ops, optimize, and
    // assert the residual re-verifies and is byte-identical on the interpreter. This hammers
    // the value renumbering + operand remapper that dead-value elimination relies on. Most
    // produced values are dead, so the optimizer should usually shrink the block.
    const NONTRAP: [BinOp; 11] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::ShrS,
        BinOp::ShrU,
        BinOp::Rotl,
        BinOp::Rotr,
    ];
    const UNOPS: [IntUnOp; 6] = [
        IntUnOp::Clz,
        IntUnOp::Ctz,
        IntUnOp::Popcnt,
        IntUnOp::Extend8S,
        IntUnOp::Extend16S,
        IntUnOp::Extend32S,
    ];

    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut shrunk = 0usize;

    for _ in 0..400 {
        let n = 1 + rng.below(40);
        let mut insts: Vec<Inst> = Vec::with_capacity(n);
        insts.push(Inst::ConstI64(rng.next() as i64)); // index 0 is always defined
        for i in 1..n {
            match rng.below(3) {
                0 => insts.push(Inst::ConstI64(rng.next() as i64)),
                1 => insts.push(Inst::IntBin {
                    ty: IntTy::I64,
                    op: NONTRAP[rng.below(NONTRAP.len())],
                    a: rng.below(i) as u32,
                    b: rng.below(i) as u32,
                }),
                _ => insts.push(Inst::IntUn {
                    ty: IntTy::I64,
                    op: UNOPS[rng.below(UNOPS.len())],
                    a: rng.below(i) as u32,
                }),
            }
        }
        let ret = rng.below(insts.len()) as u32;
        let m = single_block_fn(vec![], vec![ValType::I64], insts, ret);

        let before = m.funcs[0].blocks[0].insts.len();
        let opt = check_equiv(&m, &[vec![]]);
        let after = opt.funcs[0].blocks[0].insts.len();
        assert!(after <= before, "optimizer must never grow the block");
        if after < before {
            shrunk += 1;
        }
    }

    assert!(
        shrunk > 0,
        "dead-value elimination should fire on random DAGs"
    );
}

// ----- copy propagation + algebraic identities -----

const XS: [i64; 6] = [0, 1, -1, 7, -123_456, i64::MIN];
fn x_argsets() -> Vec<Vec<Value>> {
    XS.iter().map(|&x| vec![Value::I64(x)]).collect()
}

/// `f(x) = bin(x, k)` (or `bin(k, x)` when `const_first`), one i64 binary op against a constant.
fn one_bin(op: BinOp, k: i64, const_first: bool) -> Module {
    let (a, b) = if const_first { (1, 0) } else { (0, 1) };
    single_block_fn(
        vec![ValType::I64],
        vec![ValType::I64],
        vec![
            Inst::ConstI64(k),
            Inst::IntBin {
                ty: IntTy::I64,
                op,
                a,
                b,
            },
        ],
        2,
    )
}

/// The identity forwarded to the parameter: equivalent to the original *and* fully simplified away
/// (the constant and the op are both dead and removed, so the block just returns the param).
fn assert_forwards_to_param(m: &Module) {
    let opt = check_equiv(m, &x_argsets());
    let b = &opt.funcs[0].blocks[0];
    assert!(
        b.insts.is_empty(),
        "identity should be forwarded away, got {:?}",
        b.insts
    );
    assert_eq!(
        b.term,
        Terminator::Return(vec![0]),
        "should return the param"
    );
}

/// The identity folded to a constant: equivalent to the original, with no arithmetic op left.
fn assert_folds_to_const(m: &Module, expect: i64) {
    let opt = check_equiv(m, &x_argsets());
    let b = &opt.funcs[0].blocks[0];
    assert!(
        !b.insts
            .iter()
            .any(|i| matches!(i, Inst::IntBin { .. } | Inst::Select { .. })),
        "absorbing identity should leave no arithmetic op, got {:?}",
        b.insts
    );
    assert_eq!(run(&opt, &[Value::I64(0)]), Ok(vec![Value::I64(expect)]));
}

#[test]
fn forwards_arithmetic_identities() {
    use BinOp::*;
    for (op, k, const_first) in [
        (Add, 0, false),
        (Add, 0, true),
        (Sub, 0, false),
        (Mul, 1, false),
        (Mul, 1, true),
        (Or, 0, false),
        (Or, 0, true),
        (And, -1, false),
        (And, -1, true),
        (Xor, 0, false),
        (Xor, 0, true),
        (Shl, 0, false),
        (ShrS, 0, false),
        (ShrU, 0, false),
        (Rotl, 0, false),
        (Rotr, 0, false),
    ] {
        assert_forwards_to_param(&one_bin(op, k, const_first));
    }
}

#[test]
fn folds_absorbing_identities() {
    use BinOp::*;
    assert_folds_to_const(&one_bin(Mul, 0, false), 0); // x * 0
    assert_folds_to_const(&one_bin(Mul, 0, true), 0); // 0 * x
    assert_folds_to_const(&one_bin(And, 0, false), 0); // x & 0
    assert_folds_to_const(&one_bin(Or, -1, false), -1); // x | -1
    assert_folds_to_const(&one_bin(RemS, 1, false), 0); // x % 1
    assert_folds_to_const(&one_bin(RemU, 1, false), 0); // x %u 1
}

#[test]
fn self_cancelling_identities() {
    // Both operands are the same value (index 0).
    let selfop = |op| {
        single_block_fn(
            vec![ValType::I64],
            vec![ValType::I64],
            vec![Inst::IntBin {
                ty: IntTy::I64,
                op,
                a: 0,
                b: 0,
            }],
            1,
        )
    };
    assert_forwards_to_param(&selfop(BinOp::And)); // x & x -> x
    assert_forwards_to_param(&selfop(BinOp::Or)); // x | x -> x
    assert_folds_to_const(&selfop(BinOp::Sub), 0); // x - x -> 0
    assert_folds_to_const(&selfop(BinOp::Xor), 0); // x ^ x -> 0
}

#[test]
fn select_forwards_chosen_dynamic_operand() {
    // select(const cond, a, b) forwards the chosen operand even when it isn't a constant — the
    // case the old optimizer (which folded `select` only when the chosen value was constant) missed.
    let sel = |cond: i32, expect_param: u32| {
        let m = single_block_fn(
            vec![ValType::I64, ValType::I64],
            vec![ValType::I64],
            vec![
                Inst::ConstI32(cond),
                Inst::Select {
                    cond: 2,
                    a: 0,
                    b: 1,
                },
            ],
            3,
        );
        let opt = check_equiv(
            &m,
            &[
                vec![Value::I64(11), Value::I64(22)],
                vec![Value::I64(-1), Value::I64(5)],
            ],
        );
        let b = &opt.funcs[0].blocks[0];
        assert!(
            !b.insts.iter().any(|i| matches!(i, Inst::Select { .. })),
            "select should be forwarded away"
        );
        assert_eq!(b.term, Terminator::Return(vec![expect_param]));
    };
    sel(1, 0); // true  -> a (param 0)
    sel(0, 1); // false -> b (param 1)
}
