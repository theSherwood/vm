//! Spec for loop-invariant code motion (OPT.md Phase 2). A loop-invariant pure computation must
//! (1) still produce identical results on the reference interpreter, and (2) end up **outside** every
//! loop (its block is not in a cyclic SCC of the optimized function). A loop-*variant* computation
//! must stay in the loop.

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use svm_opt::cfg::Cfg;
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        types: vec![],
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        impl_exports: vec![],
        debug_info: None,
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

fn bin(op: BinOp, a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op,
        a,
        b,
    }
}

/// Is any instruction matching `pred` located in a block that is part of a loop (a cyclic SCC)?
fn matches_inside_loop<F: Fn(&Inst) -> bool>(f: &Func, pred: F) -> bool {
    let cfg = Cfg::new(&f.blocks);
    let mut in_loop = vec![false; f.blocks.len()];
    for comp in cfg.sccs() {
        let cyclic =
            comp.len() > 1 || (comp.len() == 1 && cfg.succs[comp[0] as usize].contains(&comp[0]));
        if cyclic {
            for b in comp {
                in_loop[b as usize] = true;
            }
        }
    }
    f.blocks
        .iter()
        .enumerate()
        .any(|(b, blk)| in_loop[b] && blk.insts.iter().any(&pred))
}

/// A counted loop that accumulates `body(a, b)` each iteration.
///
/// b0(n, a, b): br b1(n, 0, a, b)
/// b1(i, acc, a', b'): body ; acc2 = acc + body ; i2 = i - 1 ; if i2 { loop } else exit(acc2)
/// b2(r): return r
/// `body` is built from the four block-1 params (i=0, acc=1, a'=2, b'=3).
fn loop_module(body: Inst) -> Module {
    module(Func {
        params: vec![ValType::I32, ValType::I32, ValType::I32], // n, a, b
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32, ValType::I32], // n, a, b
                insts: vec![Inst::ConstI32(0)],                         // v3 = acc0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 3, 1, 2], // (i=n, acc=0, a'=a, b'=b)
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32], // i, acc, a', b'
                insts: vec![
                    body,                  // v4 = body(...)
                    bin(BinOp::Add, 1, 4), // v5 = acc + body
                    Inst::ConstI32(1),     // v6
                    bin(BinOp::Sub, 0, 6), // v7 = i - 1
                ],
                term: Terminator::BrIf {
                    cond: 7,
                    then_blk: 1,
                    then_args: vec![7, 5, 2, 3], // loop with (i-1, acc2, a', b')
                    else_blk: 2,
                    else_args: vec![5], // exit with acc2
                },
            },
            Block {
                params: vec![ValType::I32], // r
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    })
}

fn check_equiv(m: &Module) -> Module {
    verify_module(m).expect("verifies");
    let opt = optimize_module(m);
    verify_module(&opt).expect("optimized (hoisted) re-verifies");
    // n = 1,2,5 terminate; results must match exactly.
    for n in [1i32, 2, 5, 8] {
        for &(a, b) in &[(3i32, 4i32), (0, 9), (-2, 7)] {
            let args = [Value::I32(n), Value::I32(a), Value::I32(b)];
            assert_eq!(
                run(m, &args),
                run(&opt, &args),
                "divergence at n={n} a={a} b={b}"
            );
        }
    }
    opt
}

#[test]
fn invariant_multiply_is_hoisted_out_of_the_loop() {
    // body = a' * b'  — loop-invariant (a', b' are passed through unchanged every iteration).
    let m = loop_module(bin(BinOp::Mul, 2, 3));
    assert!(
        matches_inside_loop(&m.funcs[0], |i| matches!(
            i,
            Inst::IntBin { op: BinOp::Mul, .. }
        )),
        "precondition: the multiply starts inside the loop"
    );
    let opt = check_equiv(&m);
    assert!(
        !matches_inside_loop(&opt.funcs[0], |i| matches!(
            i,
            Inst::IntBin { op: BinOp::Mul, .. }
        )),
        "LICM should hoist the invariant multiply out of the loop"
    );
}

#[test]
fn variant_computation_stays_in_the_loop() {
    // body = i * a'  — depends on the loop counter `i`, so it is NOT invariant and must stay.
    let m = loop_module(bin(BinOp::Mul, 0, 2));
    let opt = check_equiv(&m);
    assert!(
        matches_inside_loop(&opt.funcs[0], |i| matches!(
            i,
            Inst::IntBin { op: BinOp::Mul, .. }
        )),
        "a loop-variant multiply must not be hoisted"
    );
}

#[test]
fn an_invariant_bare_constant_is_not_hoisted() {
    // body = i * a' (variant, stays). The loop also holds `ConstI32(1)` (the `i - 1` decrement): it is
    // loop-invariant, but a constant is free to recompute — hoisting it would only thread a parameter
    // around the loop (pure overhead, DESIGN §hoist cost model). So the constant must stay in the loop
    // alongside the variant decrement that uses it, not be lifted to the preheader.
    let m = loop_module(bin(BinOp::Mul, 0, 2));
    let opt = check_equiv(&m);
    assert!(
        matches_inside_loop(&opt.funcs[0], |i| matches!(i, Inst::ConstI32(_))),
        "an invariant constant used only by loop-variant code must not be hoisted out"
    );
}

#[test]
fn an_invariant_op_over_a_constant_is_still_hoisted() {
    // b1 computes `k = 100; t = a' + k` — loop-invariant (a' is passed through unchanged). The add is
    // hoisted out even though one operand is a loop-body constant: the constant is rematerialized in
    // the preheader (not threaded), so the add can still move. The add must end up outside the loop and
    // results must be unchanged.
    let m = module(Func {
        params: vec![ValType::I32, ValType::I32], // n, a
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32], // n, a
                insts: vec![Inst::ConstI32(0)],           // v2 = acc0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 2, 1], // (i=n, acc=0, a'=a)
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32, ValType::I32], // i, acc, a'
                insts: vec![
                    Inst::ConstI32(100),   // v3 = k (loop-body constant)
                    bin(BinOp::Add, 2, 3), // v4 = a' + k  (invariant)
                    bin(BinOp::Add, 1, 4), // v5 = acc + t
                    Inst::ConstI32(1),     // v6
                    bin(BinOp::Sub, 0, 6), // v7 = i - 1
                ],
                term: Terminator::BrIf {
                    cond: 7,
                    then_blk: 1,
                    then_args: vec![7, 5, 2], // (i-1, acc2, a')
                    else_blk: 2,
                    else_args: vec![5],
                },
            },
            Block {
                params: vec![ValType::I32], // r
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    });
    verify_module(&m).expect("verifies");
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    for n in [1i32, 2, 5, 8] {
        for a in [3i32, 0, -2] {
            let args = [Value::I32(n), Value::I32(a)];
            assert_eq!(
                run(&m, &args),
                run(&opt, &args),
                "divergence at n={n} a={a}"
            );
        }
    }
    // The invariant `a' + 100` is hoisted, and its constant operand `100` is rematerialized in the
    // preheader — so `ConstI32(100)` leaves the loop entirely (it still exists out-of-loop, since `a'`
    // is a runtime value the add can't be folded away).
    let is_100 = |i: &Inst| matches!(i, Inst::ConstI32(100));
    assert!(
        !matches_inside_loop(&opt.funcs[0], is_100),
        "the invariant add's constant operand must be rematerialized out of the loop"
    );
    assert!(
        opt.funcs[0]
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(is_100),
        "the constant is rematerialized in the preheader, not dropped"
    );
}
